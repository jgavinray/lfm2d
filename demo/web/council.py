# SPDX-License-Identifier: MIT
# Ported from the megakernel council, council/server.py (~/src/megakernel-qwen38-flashnext-strixhalo, MIT,
# 2026-10-03), onto lfm2d's held contexts and multi-context opinion reads (docs/integration.md invariants 16-19).
"""The council: several held contexts judge each proposed agent action, as a client of an lfm2d daemon.

demo/web/server.py mounts this module under /council/api/* and serves its page (static/council.html) at /council. It
starts on the first request there: the other demos never pay for it. Standard library only.

On the left, tabs: one per context (default three, from council_scenario.py: Memory, User, Session; at most 8, the
most one read takes). Each tab is a small chat: a system turn (the scenario's REVIEWER framing, the tab's name and
preamble), then messages the user adds, edits and deletes, and an ask box that runs System 2 in that tab (a streamed
POST /v1/chat; the reply joins the tab). Each tab's head is held on the daemon as a pinned context built from
messages (POST /v1/contexts {system, messages, pin: true}); the daemon refuses control-token text in any turn rather
than escaping it, and that refusal is shown to the page. After any change the tab is pinned again and its old context
deleted, unless another tab holds the same id: the id is the content, and a delete is not reference-counted.

On the right, decisions: an action (typed, or one of the scenario's) is ONE POST /v1/opinion with `contexts` = the
included tabs' contexts in tab order, the current spec, and the spec's verdict question, pooled by the daemon (linear
| loglinear, uniform | mass weights). This module recomputes the daemon's pool from the reads (council_pool.py, the
same operations as lfm2d/src/pool.rs) and stops the job loudly on any difference, then computes what the page shows:
both pools (the ternary plot's two stars), the pooled verdict (the top option, ties to the earlier one: the daemon
never picks), and leave-one-out.

Specs: two, consumer-owned, in static/ (uploaded at boot, content-addressed): council-verdict-v1 asks the verdict
cold, at the first slot, like the megakernel's letters; council-describe-v1 has each context describe the action first
and then reads the verdict, so every description comes from inside its own context. Field and option names come from
GET /v1/opinion/specs, never from this file; the two specs must ask the same options. The options run from routine to
loudest, so the last one is the louder ask the page alarms on.

Backfill: when the set of contexts changes (an edit, an ask's reply, an include toggle, a tab added or removed) or the
spec does, the last BACKFILL_K decisions are re-read under the new set, one read each, and each card whose pooled
verdict changed is marked with the flip (old -> new) and what caused it. A pool change re-pools the stored reads (no
read). Replay re-reads one decision under the same contexts and spec and compares the bits: a repeated read is
identical (invariant 17), so a mismatch is a bug, shown as one.

Restarts: a daemon restart forgets every context and uploaded spec. A read that meets a 404 for either pins every tab
again (or uploads the specs again) and retries once; the ids are the content, so they come back the same.

Ask: /v1/chat never continues from a held context and never takes an assistant turn as text (invariant 16). So an ask
starts a fresh chat from the tab's system turn and its messages (all user turns) plus the question, and when the tab
already holds a reply from an earlier ask, the ask continues that chat `from` its checkpoint, but only while the tab
still holds exactly what that chat held (later notes are appended as user turns). A tab whose reply the daemon can no
longer continue (edited since, evicted, restarted) refuses the ask with a 400 that says so: delete the reply, or add
the question as a note.

Trust: an action is the read's `state.input`. The daemon refuses control-token text in it (400) instead of escaping
it, so this module needs no denylist; the refusal goes to the page as the daemon wrote it.

One engine thread owns the daemon calls; every change runs there in order. The page gets server-sent events.

  GET  /council/api/state                 the snapshot
  GET  /council/api/events                SSE: hello (the snapshot), then status, tab, tab_removed, token, ask_done,
                                          aborted, decision, backfill, replay, reset, spec, error, repinned
  POST /council/api/decide {action}       one decision, answered when read
  POST /council/api/scenario              queue the scripted actions (202)
  POST /council/api/pool {method?, weights?}   re-pools every decision
  POST /council/api/spec {spec}           switch spec (a file name from the snapshot's specs); a backfill
  POST /council/api/tabs {name, preamble?, color?}                  add a tab (pinned, included)
  POST /council/api/tabs/{id} {name?, preamble?, color?, include?}  change a tab
  POST /council/api/tabs/{id}/remove
  POST /council/api/tabs/{id}/messages {role, content}              add a message
  POST /council/api/tabs/{id}/messages/{i} {content}                edit one
  POST /council/api/tabs/{id}/messages/{i}/remove
  POST /council/api/tabs/{id}/ask {text}  System 2 in the tab (202; tokens stream as events)
  POST /council/api/decisions/{id}/replay re-read under the same contexts and spec; {match}
  POST /council/api/stop                  stop an ask
  POST /council/api/reset                 back to the seed tabs and the first spec, no decisions
"""
from __future__ import annotations

import itertools
import json
import math
import queue
import re
import threading
import time
import traceback
import urllib.error
import urllib.request
from http import HTTPStatus
from pathlib import Path

import council_pool
import council_scenario as scenario

STATIC = Path(__file__).resolve().parent / "static"
SPEC_FILES = ["council-verdict-v1.json", "council-describe-v1.json"]
MAX_TABS = 8  # /v1/opinion reads after 1 to 8 contexts
BACKFILL_K = 12
MAX_BODY = 64 * 1024
MAX_ACTION = 2000
MAX_NAME = 32
MAX_TEXT = 8000
MAX_MESSAGES = 64
MAX_DECISIONS = 500
HISTORY = 8  # earlier reads kept per decision
ASK_TOKENS = 2048
ASK_TIMEOUT_MS = 600_000
READ_TIMEOUT_MS = 120_000
WAIT_S = 900.0  # a request that waits for its job (behind at most an ask's reply)
BOOT_RETRY_S = 5.0
MIN_MASS = 0.5  # under this much raw mass on the options, the page flags a read (as tail.html does)
# One per tab: each is its identity everywhere on the page. None is green, amber or red, the verdict colors.
PALETTE = ["#22e4ff", "#ff4fd8", "#a98bff", "#4d9dff", "#e6e6f0", "#ff8fb8", "#d4a8ff", "#9ff3ff"]
TRUST = ("An action is the read's input, rendered after each context. The daemon refuses control-token text in it, "
         "and in every tab's turns, rather than escaping it (a 400 shown here as it came), so no action or tab text "
         "can forge a chat turn.")
_COLOR = re.compile(r"#[0-9a-fA-F]{6}")
# A read naming a context the daemon no longer holds (a 404 that says so; since f6a10a6 "no chat checkpoint or
# held context", earlier "no held context").
POOLED_KEYS = ("probs", "agree", "spread", "leave_one_out", "weights")
GONE = ("no chat checkpoint or held context", "no held context", "no chat checkpoint")


class Refused(ValueError):
    """A request this module won't take (400); nothing changed."""


class Missing(LookupError):
    """No such tab or decision (404)."""


class ApiError(Exception):
    """The daemon answered with an error status."""

    def __init__(self, status: int, message: str):
        super().__init__(f"{status} {message}")
        self.status, self.message = status, message


class Unreachable(Exception):
    """The daemon did not answer at all."""


class Daemon:
    """A small JSON client of one lfm2d daemon."""

    def __init__(self, base: str, timeout: float = 180.0):
        self.base, self.timeout = base.rstrip("/"), timeout

    def _open(self, method: str, path: str, data: bytes | None, timeout: float):
        req = urllib.request.Request(self.base + path, data, method=method,
                                     headers={"content-type": "application/json"})
        try:
            return urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as e:
            with e:
                raw = e.read()
            try:
                msg = json.loads(raw)["error"]["message"]
            except (ValueError, KeyError, TypeError):
                msg = raw.decode("utf-8", "replace")[:500]
            raise ApiError(e.code, msg) from None
        except (urllib.error.URLError, OSError) as e:
            raise Unreachable(f"{method} {path}: {e}") from None

    def call(self, method: str, path: str, body=None, raw: bytes | None = None):
        data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
        with self._open(method, path, data, self.timeout) as r:
            return r.status, json.loads(r.read() or b"null")

    def get(self, path: str):
        return self.call("GET", path)[1]

    def post(self, path: str, body):
        return self.call("POST", path, body)[1]

    def delete(self, path: str):
        return self.call("DELETE", path)[1]

    def upload(self, path: str, raw: bytes):
        return self.call("POST", path, raw=raw)

    def stream(self, path: str, body, stop: threading.Event):
        """POST and yield (event, data) off a server-sent-event answer; leaving the loop (or `stop`) hangs up, and
        the daemon cancels a chat turn whose client went."""
        with self._open("POST", path, json.dumps(body).encode(), 60.0) as r:
            event, data = "message", []
            while not stop.is_set():
                line = r.readline()
                if not line:
                    return
                line = line.decode("utf-8").rstrip("\r\n")
                if line == "":
                    if data:
                        yield event, json.loads("\n".join(data))
                    event, data = "message", []
                elif line.startswith(":"):
                    continue  # keep-alive
                else:
                    k, _, v = line.partition(":")
                    v = v[1:] if v.startswith(" ") else v
                    if k == "event":
                        event = v
                    elif k == "data":
                        data.append(v)


class Job:
    def __init__(self, kind: str, args: dict, wait: bool):
        self.kind, self.args = kind, args
        self.done = threading.Event() if wait else None
        self.result = None
        self.error: BaseException | None = None


def _check_action(action) -> str:
    if not isinstance(action, str) or not action.strip():
        raise Refused("action must be a non-empty string")
    if len(action) > MAX_ACTION:
        raise Refused(f"the action is too long ({len(action)} characters, at most {MAX_ACTION})")
    return action.strip()


def _check_name(name) -> str:
    if not isinstance(name, str) or not name.strip():
        raise Refused("name must be a non-empty string")
    if len(name.strip()) > MAX_NAME:
        raise Refused(f"name is too long (at most {MAX_NAME} characters)")
    return name.strip()


def _check_text(value, what: str, limit: int = MAX_TEXT) -> str:
    if not isinstance(value, str) or not value.strip():
        raise Refused(f"{what} must be a non-empty string")
    if len(value) > limit:
        raise Refused(f"{what} is too long (at most {limit} characters)")
    return value


def _fields(body: dict, required: set, optional: set = frozenset()) -> dict:
    unknown = set(body) - required - optional
    if unknown:
        raise Refused(f"unknown fields {sorted(unknown)}")
    missing = required - set(body)
    if missing:
        raise Refused(f"{', '.join(sorted(missing))} is required")
    return body


def _refused_by_daemon(e: ApiError, what: str) -> Refused:
    return Refused(f"the daemon refused {what}: {e.message}")


class Council:
    def __init__(self, daemon: Daemon, seeds: list[dict] | None = None, actions: list[dict] | None = None,
                 spec_files: list[str] | None = None, static: Path = STATIC):
        self.daemon = daemon
        self.seeds = scenario.TABS if seeds is None else seeds
        self.actions = scenario.ACTIONS if actions is None else actions
        self.spec_files = SPEC_FILES if spec_files is None else spec_files
        self.static = static
        self.lock = threading.RLock()
        self.gate = threading.Lock()
        self.jobs: queue.Queue[Job] = queue.Queue()
        self.subs: list[queue.Queue] = []
        self.status = {"phase": "starting", "detail": "not started"}
        self.booted = False
        self.started = False
        self.model: dict | None = None
        self.specs: dict[str, dict] = {}  # file -> {file, id, raw, field, options, describe, input_label}
        self.spec = self.spec_files[0]
        self.options: list[str] = []
        self.tabs: list[dict] = []
        self.decisions: list[dict] = []
        self.pool = {"method": "linear", "weights": "uniform"}
        self.causes: list[dict] = []  # context or spec changes since the last backfill
        self.tab_seq = itertools.count(1)
        self.dec_seq = itertools.count(1)
        self.stop_flag = threading.Event()
        self.asking = False

    # --- events -------------------------------------------------------------------

    def emit(self, ev: dict) -> None:
        with self.lock:
            for q in list(self.subs):
                try:
                    q.put_nowait(ev)
                except queue.Full:  # this browser stopped reading: disconnect it rather than buffer without end
                    q.dropped = True
                    self.subs.remove(q)

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=4096)
        q.dropped = False
        with self.lock:
            self.subs.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self.lock:
            if q in self.subs:
                self.subs.remove(q)

    def set_status(self, phase: str, detail: str = "") -> None:
        self.status = {"phase": phase, "detail": detail}
        self.emit({"type": "status", **self.status})

    # --- jobs ---------------------------------------------------------------------

    def start(self) -> None:
        """Start the engine thread once (the first request under /council/api does)."""
        with self.gate:
            if self.started:
                return
            self.started = True
        threading.Thread(target=self.run, name="council", daemon=True).start()

    def submit(self, kind: str, wait: bool = False, **args) -> Job:
        job = Job(kind, args, wait)
        with self.gate:
            if self.status["phase"] == "failed":
                job.error = RuntimeError(f"the council is down: {self.status['detail']}")
                return job
            self.jobs.put(job)
        if wait and not job.done.wait(WAIT_S):
            job.error = TimeoutError(f"{kind} did not finish in {WAIT_S:g} s")
        return job

    def quiet(self) -> bool:
        """Booted, nothing queued and nothing running."""
        return self.booted and self.jobs.unfinished_tasks == 0

    def run(self) -> None:
        while True:
            try:
                self.boot()
                break
            except (Unreachable, ApiError) as e:
                if isinstance(e, ApiError) and e.status not in (502, 503, 504):
                    self.fail(e)
                    return
                self.set_status("waiting", f"the daemon is not answering yet ({e}); retrying")
                time.sleep(BOOT_RETRY_S)
            except Exception as e:  # noqa: BLE001 -- a boot that can't work: fail loudly, once
                self.fail(e)
                return
        while True:
            job = self.jobs.get()
            try:
                job.result = getattr(self, f"do_{job.kind}")(**job.args)
            except Exception as e:  # noqa: BLE001 -- the job's error goes to its waiter and to the page
                if not isinstance(e, (Refused, Missing)):
                    traceback.print_exc()
                job.error = e
                if not job.done:
                    self.emit({"type": "error", "job": job.kind, "message": f"{type(e).__name__}: {e}"})
            finally:
                if self.jobs.qsize() == 0:
                    self.set_status("idle")
                if job.done:
                    job.done.set()
                self.jobs.task_done()

    def fail(self, e: BaseException) -> None:
        traceback.print_exc()
        with self.gate:
            self.set_status("failed", f"{type(e).__name__}: {e}")
        while True:  # nothing will run what is queued: fail it loudly
            try:
                job = self.jobs.get_nowait()
            except queue.Empty:
                return
            job.error = RuntimeError(f"the council is down: {self.status['detail']}")
            self.emit({"type": "error", "job": job.kind, "message": str(job.error)})
            if job.done:
                job.done.set()
            self.jobs.task_done()

    # --- boot ---------------------------------------------------------------------

    def boot(self) -> None:
        """The daemon answers; the specs are loaded and ask the same options; then pin the seed tabs."""
        self.set_status("starting", "asking the daemon")
        info = self.daemon.get("/v1/adjudicator")
        self.model = {k: info.get(k) for k in ("model_id", "weight_hash", "backend", "device")}
        self.upload_specs()
        first = self.specs[self.spec_files[0]]
        for f in self.spec_files[1:]:
            if self.specs[f]["options"] != first["options"]:
                raise RuntimeError(f"{f} asks {self.specs[f]['options']}, {self.spec_files[0]} asks "
                                   f"{first['options']}: switching specs would change what a verdict is")
        self.options = list(first["options"])
        self.seed_tabs()
        self.booted = True
        self.set_status("idle")

    def upload_specs(self) -> None:
        """Upload every spec (201 new, 200 already loaded) and read its question off the menu: the LAST choice field
        in emission order, so the fields before it are the description."""
        ids = {}
        for f in self.spec_files:
            raw = (self.static / f).read_bytes()
            status, entry = self.daemon.upload("/v1/opinion/specs", raw)
            if status not in (200, 201):
                raise RuntimeError(f"uploading {f} answered {status}")
            ids[f] = entry["id"]
        menu = {e["id"]: e for e in self.daemon.get("/v1/opinion/specs")}
        for f, sid in ids.items():
            entry = menu.get(sid)
            if entry is None:
                raise RuntimeError(f"{f} was uploaded as {sid} but the menu does not list it")
            choices = [i for i, x in enumerate(entry["fields"]) if x["kind"] == "choice"]
            if not choices:
                raise RuntimeError(f"{f} has no choice field to ask")
            q = choices[-1]
            if len(entry["fields"][q]["options"]) < 2:
                raise RuntimeError(f"{f}: field {entry['fields'][q]['field']!r} has fewer than two options")
            self.specs[f] = {"file": f, "id": sid, "field": entry["fields"][q]["field"],
                             "options": list(entry["fields"][q]["options"]), "input_label": entry["input_label"],
                             "describe": [x["field"] for x in entry["fields"][:q]]}

    # --- tabs ---------------------------------------------------------------------

    def seed_tabs(self) -> None:
        for k, s in enumerate(self.seeds):
            tab = {"id": f"t{next(self.tab_seq)}", "name": s["name"], "color": s.get("color", PALETTE[k % len(PALETTE)]),
                   "include": s.get("include", True), "preamble": s.get("preamble", ""),
                   "messages": [dict(m) for m in s.get("messages", [])],
                   "context": None, "n_tokens": 0, "pin_ms": 0.0, "chat": None}
            self.pin(tab)
            with self.lock:
                self.tabs.append(tab)
            self.emit({"type": "tab", "tab": self.tab_view(tab)})

    def head(self, tab: dict) -> tuple[str, list[dict]]:
        """A tab's head: the system turn (the reviewer framing, the tab's name and preamble) and its messages, an
        ask's reply with the reasoning it was generated with."""
        system = scenario.REVIEWER + f"\n\nTHIS SOURCE: {tab['name']}"
        if tab["preamble"].strip():
            system += f"\n{tab['preamble'].strip()}"
        msgs = []
        for m in tab["messages"]:
            x = {"role": m["role"], "content": m["content"]}
            if m["role"] == "assistant" and m.get("thinking") is not None:
                x["thinking"] = m["thinking"]
            msgs.append(x)
        return system, msgs

    def pin(self, tab: dict) -> str | None:
        """Hold the tab's head pinned and point the tab at it; the old context id, for the caller to `retire` once
        its change is committed (a failed DELETE must not fail a change that already happened)."""
        self.set_status("pinning", tab["name"])
        system, msgs = self.head(tab)
        try:
            ctx = self.daemon.post("/v1/contexts", {"system": system, "messages": msgs, "pin": True})
        except ApiError as e:
            if e.status == 400:
                raise _refused_by_daemon(e, f"tab {tab['name']!r}") from None
            raise
        if not ctx["pinned"]:
            raise RuntimeError(f"the daemon did not pin the context of tab {tab['name']!r}")
        old = tab["context"]
        with self.lock:
            tab["context"], tab["n_tokens"] = ctx["id"], ctx["n_tokens"]
            tab["pin_ms"] = round(ctx.get("prefill_ms", 0.0), 1)
            tab["cached_tokens"] = ctx.get("cached_tokens", 0)
        return old if old != ctx["id"] else None

    def repin_all(self) -> None:
        """Pin every tab's head again after the daemon lost them (a restart forgets every context)."""
        for tab in self.tabs:
            old = self.pin(tab)
            self.emit({"type": "tab", "tab": self.tab_view(tab)})
            self.retire(old)
        self.emit({"type": "repinned", "tabs": len(self.tabs),
                   "message": f"the daemon no longer held the tabs' contexts (a restart?): pinned {len(self.tabs)} "
                              f"again"})

    def retire(self, context: str | None) -> None:
        """Delete a context no tab holds any longer; a DELETE that fails is reported, not raised: the change that
        dropped the context stands."""
        if not context:
            return
        try:
            self.release(context)
        except Exception as e:  # noqa: BLE001 -- reported to the page with the id, so it can be freed by hand
            traceback.print_exc()
            self.emit({"type": "error", "job": "free", "message": f"context {context} stays pinned on the daemon "
                       f"(DELETE failed: {type(e).__name__}: {e}); delete it by hand or restart the daemon"})

    def release(self, context: str) -> None:
        """A delete is not reference-counted: never delete an id another tab still holds. A 404 is already the goal."""
        if any(t["context"] == context for t in self.tabs):
            return
        try:
            self.daemon.delete(f"/v1/contexts/{context}")
        except ApiError as e:
            if e.status != 404:
                raise

    def tab(self, tid: str) -> dict:
        for t in self.tabs:
            if t["id"] == tid:
                return t
        raise Missing(f"no tab {tid!r}")

    def tab_view(self, tab: dict) -> dict:
        v = json.loads(json.dumps(tab))
        v["chat"] = bool(tab.get("chat"))
        return v

    def changed(self, what: str, tab: dict | None = None, **extra) -> None:
        """A change to the contexts' set or the spec: note its cause and queue a backfill (it runs after this job)."""
        cause = {"what": what, "t": time.time(), **extra}
        if tab is not None:
            cause.update({"tab": tab["id"], "name": tab["name"], "color": tab["color"]})
        with self.lock:
            self.causes.append(cause)
        self.submit("backfill")

    def do_tab_add(self, name: str, preamble: str = "", color: str | None = None) -> dict:
        if len(self.tabs) >= MAX_TABS:
            raise Refused(f"at most {MAX_TABS} tabs: a read takes at most {MAX_TABS} contexts")
        used = {t["color"] for t in self.tabs}
        color = color or next((c for c in PALETTE if c not in used), PALETTE[len(self.tabs) % len(PALETTE)])
        tab = {"id": f"t{next(self.tab_seq)}", "name": name, "color": color, "include": True, "preamble": preamble,
               "messages": [], "context": None, "n_tokens": 0, "pin_ms": 0.0, "chat": None}
        self.pin(tab)
        with self.lock:
            self.tabs.append(tab)
        self.emit({"type": "tab", "tab": self.tab_view(tab)})
        self.changed("add", tab)
        return self.tab_view(tab)

    def do_tab_update(self, tid: str, **kw) -> dict:
        tab = self.tab(tid)
        repin = any(k in kw and kw[k] != tab[k] for k in ("name", "preamble"))
        include = "include" in kw and kw["include"] != tab["include"]
        saved = dict(tab)
        with self.lock:
            tab.update(kw)
        old = None
        if repin:
            try:
                old = self.pin(tab)
            except BaseException:
                with self.lock:
                    tab.clear()
                    tab.update(saved)
                raise
        self.emit({"type": "tab", "tab": self.tab_view(tab)})
        self.retire(old)
        if repin and tab["include"]:
            self.changed("rename" if "name" in kw and kw["name"] != saved["name"] else "preamble", tab)
        elif include:
            self.changed("include" if tab["include"] else "exclude", tab)
        return self.tab_view(tab)

    def do_tab_remove(self, tid: str) -> dict:
        tab = self.tab(tid)
        with self.lock:
            self.tabs.remove(tab)
        self.emit({"type": "tab_removed", "id": tid})
        self.retire(tab["context"])
        if tab["include"]:
            self.changed("remove", tab)
        return {"removed": tid}

    def edit_messages(self, tab: dict, change, what: str) -> dict:
        """Apply `change` to the tab's messages, pin again, and restore them if the pin fails."""
        saved = [dict(m) for m in tab["messages"]]
        with self.lock:
            change(tab["messages"])
        try:
            old = self.pin(tab)
        except BaseException:
            with self.lock:
                tab["messages"][:] = saved
            raise
        self.emit({"type": "tab", "tab": self.tab_view(tab)})
        self.retire(old)
        if tab["include"]:
            self.changed(what, tab)
        return self.tab_view(tab)

    def do_message_add(self, tid: str, role: str, content: str) -> dict:
        tab = self.tab(tid)
        if len(tab["messages"]) >= MAX_MESSAGES:
            raise Refused(f"tab {tab['name']!r} has {MAX_MESSAGES} messages, the most a tab holds")
        return self.edit_messages(tab, lambda ms: ms.append({"role": role, "content": content}), "add message")

    def _index(self, tab: dict, i: int) -> int:
        if not 0 <= i < len(tab["messages"]):
            raise Refused(f"no message {i} in tab {tab['name']!r} (it has {len(tab['messages'])})")
        return i

    def do_message_edit(self, tid: str, index: int, content: str) -> dict:
        tab = self.tab(tid)
        i = self._index(tab, index)

        def change(ms):
            ms[i] = {"role": ms[i]["role"], "content": content}  # an edited reply drops the reasoning it no longer matches
        return self.edit_messages(tab, change, "edit")

    def do_message_remove(self, tid: str, index: int) -> dict:
        tab = self.tab(tid)
        i = self._index(tab, index)
        return self.edit_messages(tab, lambda ms: ms.pop(i), "delete message")

    # --- System 2 in a tab ----------------------------------------------------------

    def ask_route(self, tab: dict) -> dict:
        """How an ask in this tab reaches /v1/chat: a fresh chat when the tab holds no reply, else a continuation of
        the chat that wrote its replies, while the tab still holds exactly what that chat held. Refused otherwise:
        /v1/chat never takes an assistant turn as text."""
        system, msgs = self.head(tab)
        if not any(m["role"] == "assistant" for m in msgs):
            return {"system": system, "messages": msgs}
        chat = tab.get("chat")
        if chat and chat["system"] == system and msgs[:len(chat["messages"])] == chat["messages"] \
                and all(m["role"] == "user" for m in msgs[len(chat["messages"]):]):
            return {"from": chat["checkpoint"], "messages": msgs[len(chat["messages"]):]}
        raise Refused(f"tab {tab['name']!r} holds a reply that no chat on the daemon continues (it was written by "
                      f"hand, edited since, or the daemon restarted), and /v1/chat never takes a reply back as text: "
                      f"delete the reply to ask here, or add your question as a note")

    def stop(self) -> None:
        self.stop_flag.set()

    def do_ask(self, tid: str, text: str) -> dict:
        tab = self.tab(tid)
        if len(tab["messages"]) + 2 > MAX_MESSAGES:
            raise Refused(f"tab {tab['name']!r} is full ({MAX_MESSAGES} messages)")
        route = self.ask_route(tab)
        self.stop_flag.clear()
        body = {**route, "messages": route["messages"] + [{"role": "user", "content": text}],
                "max_tokens": ASK_TOKENS, "timeout_ms": ASK_TIMEOUT_MS, "stream": True}
        with self.lock:
            tab["messages"].append({"role": "user", "content": text})
        n0 = len(tab["messages"]) - 1
        self.emit({"type": "tab", "tab": self.tab_view(tab)})
        self.set_status("thinking", tab["name"])
        done, t0 = None, time.perf_counter()
        self.asking = True
        try:
            try:
                for name, data in self.daemon.stream("/v1/chat", body, self.stop_flag):
                    if name == "token":
                        self.emit({"type": "token", "tab": tid, "piece": data.get("text", "")})
                    elif name == "done":
                        done = data
                    elif name == "error":
                        err = data.get("error", {})
                        raise ApiError(data.get("status", 500), err.get("message", json.dumps(data)))
            except ApiError as e:
                if e.status == 404 and "from" in route:
                    tab["chat"] = None
                    raise Refused(f"the daemon no longer holds the chat behind tab {tab['name']!r}'s replies "
                                  f"(evicted or restarted), and /v1/chat never takes a reply back as text: delete "
                                  f"the reply to ask here") from None
                if e.status == 400:
                    raise _refused_by_daemon(e, "the question") from None
                raise
            if done is None:
                raise RuntimeError("the reply was stopped; the question was dropped")
            if done["finish_reason"] != "stop" or done.get("checkpoint") is None or done.get("content") is None:
                raise RuntimeError(f"the reply ended by {done['finish_reason']} after {done['completion_tokens']} "
                                   f"tokens; the question was dropped")
            reply = {"role": "assistant", "content": done["content"]}
            if done.get("thinking") is not None:
                reply["thinking"] = done["thinking"]
            with self.lock:
                tab["messages"].append(reply)
            old = self.pin(tab)
            system, msgs = self.head(tab)
            tab["chat"] = {"checkpoint": done["checkpoint"], "system": system, "messages": msgs}
        except BaseException as e:
            with self.lock:
                del tab["messages"][n0:]
            self.emit({"type": "tab", "tab": self.tab_view(tab)})
            self.emit({"type": "aborted", "tab": tid, "reason": str(e)})
            raise
        finally:
            self.asking = False
        s = time.perf_counter() - t0
        self.emit({"type": "tab", "tab": self.tab_view(tab)})
        self.retire(old)
        self.emit({"type": "ask_done", "tab": tid, "tokens": done["completion_tokens"], "s": round(s, 2),
                   "continued": "from" in route})
        if tab["include"]:
            self.changed("ask", tab)
        return {"tokens": done["completion_tokens"]}

    # --- reads --------------------------------------------------------------------

    def included(self) -> list[dict]:
        tabs = [t for t in self.tabs if t["include"]]
        if not tabs:
            raise Refused("no tab is included: include at least one context to decide")
        ids = [t["context"] for t in tabs]
        if len(set(ids)) != len(ids):
            raise Refused("two included tabs hold the same context (the same head); a read takes distinct contexts: "
                          "exclude one or edit it")
        return tabs

    def opinion(self, spec: dict, action: str, ids: list[str]) -> tuple[dict, float]:
        body = {"spec": spec["id"], "state": {"input": action}, "questions": [{"field": spec["field"]}],
                "contexts": ids, "pool": dict(self.pool), "timeout_ms": READ_TIMEOUT_MS}
        t0 = time.perf_counter()
        r = self.daemon.post("/v1/opinion", body)
        return r, (time.perf_counter() - t0) * 1000

    def read_one(self, action: str, tabs: list[dict]) -> dict:
        """One /v1/opinion across these tabs for this action; on a 404 for a lost context or spec, pin or upload
        again and retry once."""
        spec = self.specs[self.spec]
        ids = [t["context"] for t in tabs]
        try:
            r, wall = self.opinion(spec, action, ids)
        except ApiError as e:
            if e.status == 400:
                raise _refused_by_daemon(e, "the action") from None
            if e.status != 404:
                raise
            if any(g in e.message for g in GONE):
                self.repin_all()
            elif "no loaded spec" in e.message:
                self.upload_specs()
                spec = self.specs[self.spec]
                self.emit({"type": "repinned", "tabs": 0,
                           "message": "the daemon no longer held the specs (a restart?): uploaded them again"})
            else:
                raise
            ids = [t["context"] for t in tabs]
            r, wall = self.opinion(spec, action, ids)
        return self.record(r, wall, spec, tabs, ids)

    def record(self, r: dict, wall: float, spec: dict, tabs: list[dict], ids: list[str]) -> dict:
        """The stored read: each context's raw numbers, checked against the daemon's pool."""
        field, options = spec["field"], spec["options"]
        if r["contexts"] != ids or len(r["reads"]) != len(ids):
            raise RuntimeError(f"/v1/opinion answered {len(r['reads'])} reads for contexts {r['contexts']}, asked "
                               f"{ids}")
        per = []
        for t, x in zip(tabs, r["reads"]):
            a = x["answers"][0]
            if a["field"] != field or [o["option"] for o in a["options"]] != options:
                raise RuntimeError(f"a read answered {a['field']!r} over {[o['option'] for o in a['options']]}, "
                                   f"asked {field!r} over {options}")
            logprobs = [council_pool.f32(o["logprob"]) for o in a["options"]]
            mass = math.exp(council_pool.f32(a["sequence_mass"]))
            probs = council_pool.option_probs(logprobs)
            per.append({"tab": t["id"], "name": t["name"], "color": t["color"], "logprobs": logprobs, "mass": mass,
                        "probs": probs, "verdict": options[council_pool.argmax(probs)],
                        "described": x.get("described", []), "cache": x.get("cache"),
                        "prompt_tokens": x.get("prompt_tokens"), "cached_tokens": x.get("cached_tokens"),
                        "described_tokens": x.get("described_tokens"),
                        "ms": {k: x.get(k) for k in ("prefill_ms", "describe_ms", "read_ms")}})
        pooled = [p for p in r["pooled"] if p["field"] == field]
        if len(pooled) != 1 or pooled[0]["options"] != options:
            raise RuntimeError(f"/v1/opinion pooled {[p['field'] for p in r['pooled']]}, asked {field!r}")
        if r["pool"] != self.pool:
            raise RuntimeError(f"/v1/opinion pooled under {r['pool']}, asked {self.pool}")
        mine = council_pool.pool([p["logprobs"] for p in per], [p["mass"] for p in per], self.pool["method"],
                                 self.pool["weights"])
        # every number the daemon pooled must be ours to the bit (`weights` since the daemon added it)
        theirs = {k: pooled[0][k] for k in POOLED_KEYS if k in pooled[0]}
        if not set(POOLED_KEYS[:4]) <= set(theirs) or {k: mine[k] for k in theirs} != theirs:
            raise RuntimeError(f"the daemon pooled {theirs}; council_pool.py gives {mine} from the same reads: the "
                               f"pooling is not the one this page explains")
        read = {"spec": spec["file"], "spec_id": spec["id"], "field": field, "options": options,
                "tabs": [t["id"] for t in tabs], "contexts": ids, "per": per, "ms": round(wall, 1),
                "queue_ms": r.get("queue_ms"), "t": time.time()}
        return self.derive(read)

    def derive(self, read: dict) -> dict:
        """The pooled verdict under the current settings, both pools' stars, and leave-one-out, from the stored reads."""
        lg, ms, options = [p["logprobs"] for p in read["per"]], [p["mass"] for p in read["per"]], read["options"]
        m, w = self.pool["method"], self.pool["weights"]
        pooled = council_pool.pool(lg, ms, m, w)
        read["pool"] = dict(self.pool)
        read["pooled"] = {"probs": pooled["probs"], "agree": pooled["agree"], "spread": pooled["spread"],
                          "verdict": options[council_pool.argmax(pooled["probs"])]}
        read["stars"] = {k: council_pool.pool(lg, ms, k, w)["probs"] for k in council_pool.METHODS}
        read["loo"] = [{"tab": p["tab"], "probs": x,
                        "verdict": None if x is None else options[council_pool.argmax(x)]}
                       for p, x in zip(read["per"], pooled["leave_one_out"])]
        return read

    def do_decide(self, action: str, source: str = "typed", rules: str | None = None, note: str | None = None) -> dict:
        self.set_status("reading", action[:60])
        read = self.read_one(action, self.included())
        n = next(self.dec_seq)
        d = {"id": f"d{n}", "n": n, "action": action, "source": source, "rules": rules, "note": note, "t": time.time(),
             "read": read, "history": [], "flip": None, "replay": None}
        with self.lock:
            self.decisions.append(d)
            del self.decisions[:-MAX_DECISIONS]
        self.emit({"type": "decision", "decision": d})
        return d

    def do_backfill(self) -> dict:
        with self.lock:
            causes, self.causes = self.causes, []
        if not causes:
            return {"reread": 0}  # an earlier backfill already covered these changes
        recent = self.decisions[-BACKFILL_K:]
        try:
            tabs = self.included()
        except Refused as e:
            self.emit({"type": "backfill", "cause": causes, "reread": 0, "flips": [], "decisions": [], "note": str(e)})
            return {"reread": 0}
        ids = [t["context"] for t in tabs]
        todo = [d for d in recent if d["read"]["contexts"] != ids or d["read"]["spec"] != self.spec]
        if not todo:
            self.emit({"type": "backfill", "cause": causes, "reread": 0, "flips": [], "decisions": []})
            return {"reread": 0}
        reads = []
        try:
            for k, d in enumerate(todo):
                self.set_status("backfill", f"re-reading {k + 1} of {len(todo)} under the new contexts")
                reads.append(self.read_one(d["action"], tabs))
        except BaseException:
            with self.lock:
                self.causes[:0] = causes  # the next change's backfill re-reads these too
            raise
        flips = []
        with self.lock:
            for d, new in zip(todo, reads):
                old = d["read"]
                d["history"] = (d["history"] + [old])[-HISTORY:]
                d["read"] = new
                a, b = old["pooled"]["verdict"], new["pooled"]["verdict"]
                d["flip"] = {"from": a, "to": b, "cause": causes[-1]} if a != b else None
                d["replay"] = None
                if a != b:
                    flips.append({"id": d["id"], "from": a, "to": b})
        self.emit({"type": "backfill", "cause": causes, "reread": len(todo), "flips": flips,
                   "decisions": [json.loads(json.dumps(d)) for d in todo]})
        return {"reread": len(todo), "flips": flips}

    def do_pool(self, method: str, weights: str) -> dict:
        if method == self.pool["method"] and weights == self.pool["weights"]:
            return {"pool": dict(self.pool)}
        with self.lock:
            self.pool = {"method": method, "weights": weights}
            flips = []
            cause = {"what": "pool", "pool": dict(self.pool), "t": time.time()}
            for d in self.decisions:
                a = d["read"]["pooled"]["verdict"]
                self.derive(d["read"])
                for h in d["history"]:
                    self.derive(h)
                b = d["read"]["pooled"]["verdict"]
                d["flip"] = {"from": a, "to": b, "cause": cause} if a != b else None
                if a != b:
                    flips.append({"id": d["id"], "from": a, "to": b})
            changed = json.loads(json.dumps(self.decisions))
        self.emit({"type": "backfill", "cause": [cause], "reread": 0, "flips": flips, "decisions": changed,
                   "pool": dict(self.pool)})
        return {"pool": dict(self.pool), "flips": flips}

    def do_spec(self, spec: str) -> dict:
        if spec not in self.specs:
            raise Refused(f"spec must be one of {list(self.specs)}")
        if spec == self.spec:
            return {"spec": spec}
        with self.lock:
            self.spec = spec
        self.emit({"type": "spec", "spec": spec})
        self.changed("spec", spec=spec)
        return {"spec": spec}

    def do_replay(self, did: str) -> dict:
        d = next((x for x in self.decisions if x["id"] == did), None)
        if d is None:
            raise Missing(f"no decision {did!r}")
        tabs = self.included()
        if d["read"]["contexts"] != [t["context"] for t in tabs] or d["read"]["spec"] != self.spec:
            raise Refused("its contexts or spec changed since it was read (it is older than the backfill reaches): "
                          "replay re-reads under the same contexts and spec only")
        self.set_status("reading", f"replaying {did}")
        new = self.read_one(d["action"], tabs)
        diffs = [p["name"] for p, q in zip(d["read"]["per"], new["per"])
                 if (p["logprobs"], p["mass"], p["described"]) != (q["logprobs"], q["mass"], q["described"])]
        match = not diffs
        with self.lock:
            d["replay"] = {"match": match, "t": time.time(), "diffs": diffs, "ms": new["ms"]}
        self.emit({"type": "replay", "id": did, "match": match, "diffs": diffs, "ms": new["ms"]})
        if not match:
            self.emit({"type": "error", "job": "replay", "message": f"replay of {did} differs from its read in "
                       f"{diffs}: a repeated read is identical (invariant 17), so this is a bug"})
        return {"id": did, "match": match, "diffs": diffs}

    def do_reset(self) -> dict:
        old = [t["context"] for t in self.tabs]
        with self.lock:
            self.tabs, self.decisions, self.causes = [], [], []
            self.spec = self.spec_files[0]
        self.seed_tabs()
        self.emit({"type": "reset", **self.snapshot()})
        for c in dict.fromkeys(old):
            self.retire(c)  # a seed head that came back under the same id is held again, and stays
        return {"tabs": len(self.tabs)}

    # --- views --------------------------------------------------------------------

    def snapshot(self) -> dict:
        with self.lock:
            return json.loads(json.dumps({
                "status": self.status, "tabs": [self.tab_view(t) for t in self.tabs], "decisions": self.decisions,
                "pool": self.pool, "spec": self.spec, "options": self.options,
                "specs": [{k: s[k] for k in ("file", "id", "field", "options", "describe", "input_label")}
                          for s in self.specs.values()],
                "palette": PALETTE, "backfill_k": BACKFILL_K, "max_tabs": MAX_TABS, "max_decisions": MAX_DECISIONS,
                "min_mass": MIN_MASS, "model": self.model, "about": scenario.ABOUT, "trust": TRUST,
                "synthetic": scenario.SYNTHETIC, "actions": self.actions}))

    # --- HTTP ---------------------------------------------------------------------

    def handle(self, h, method: str) -> None:
        """Answer a request under /council/api for demo/web/server.py's handler `h`."""
        self.start()
        path = h.path.split("?")[0][len("/council/api"):]
        if method == "GET":
            if path == "/state":
                return send_json(h, self.snapshot())
            if path == "/events":
                return self.events(h)
            return send_json(h, {"error": {"message": "not found"}}, HTTPStatus.NOT_FOUND)
        try:
            b = read_body(h)
            if not self.booted and path != "/stop":
                return send_json(h, {"error": {"message": f"the council is {self.status['phase']}: "
                                                          f"{self.status['detail']}"}},
                                 HTTPStatus.SERVICE_UNAVAILABLE)
            self.route(h, path, b)
        except Refused as e:
            send_json(h, {"error": {"message": str(e)}}, HTTPStatus.BAD_REQUEST)
        except Missing as e:
            send_json(h, {"error": {"message": str(e)}}, HTTPStatus.NOT_FOUND)
        except (ApiError, Unreachable) as e:
            send_json(h, {"error": {"message": f"the daemon: {e}"}}, HTTPStatus.BAD_GATEWAY)
        except Exception as e:  # noqa: BLE001 -- a failed job: say what, keep serving
            send_json(h, {"error": {"message": f"{type(e).__name__}: {e}"}}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def events(self, h) -> None:
        q = self.subscribe()
        try:
            h.send_response(HTTPStatus.OK)
            h.send_header("content-type", "text/event-stream")
            h.send_header("cache-control", "no-store")
            h.end_headers()
            h.close_connection = True
            h.wfile.write(f"data: {json.dumps({'type': 'hello', **self.snapshot()})}\n\n".encode())
            h.wfile.flush()
            while not q.dropped:
                try:
                    ev = q.get(timeout=15)
                    h.wfile.write(f"data: {json.dumps(ev)}\n\n".encode())
                except queue.Empty:
                    h.wfile.write(b": keepalive\n\n")
                h.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            self.unsubscribe(q)

    def run_job(self, h, kind: str, **args) -> None:
        job = self.submit(kind, wait=True, **args)
        if job.error is not None:
            raise job.error
        send_json(h, job.result)

    def route(self, h, path: str, b: dict) -> None:
        if path == "/decide":
            _fields(b, {"action"})
            self.run_job(h, "decide", action=_check_action(b["action"]))
        elif path == "/scenario":
            _fields(b, set())
            for a in self.actions:
                self.submit("decide", action=_check_action(a["text"]), source="scenario", rules=a.get("rules"),
                            note=a.get("note"))
            send_json(h, {"queued": len(self.actions)}, HTTPStatus.ACCEPTED)
        elif path == "/pool":
            _fields(b, set(), {"method", "weights"})  # either left out keeps its current setting
            method, weights = b.get("method", self.pool["method"]), b.get("weights", self.pool["weights"])
            if method not in council_pool.METHODS:
                raise Refused(f"pool method must be one of {list(council_pool.METHODS)}")
            if weights not in council_pool.WEIGHT_NAMES:
                raise Refused(f"pool weights must be one of {list(council_pool.WEIGHT_NAMES)}")
            self.run_job(h, "pool", method=method, weights=weights)
        elif path == "/spec":
            _fields(b, {"spec"})
            if b["spec"] not in self.specs:
                raise Refused(f"spec must be one of {list(self.specs)}")
            self.run_job(h, "spec", spec=b["spec"])
        elif path == "/tabs":
            _fields(b, {"name"}, {"preamble", "color"})
            args = {"name": _check_name(b["name"])}
            if "preamble" in b:
                args["preamble"] = "" if b["preamble"] == "" else _check_text(b["preamble"], "preamble")
            if "color" in b:
                args["color"] = _color(b["color"])
            self.run_job(h, "tab_add", **args)
        elif path == "/stop":
            self.stop()
            send_json(h, {"ok": True})
        elif path == "/reset":
            _fields(b, set())
            self.run_job(h, "reset")
        elif m := _REPLAY.fullmatch(path):
            _fields(b, set())
            self.run_job(h, "replay", did=m.group(1))
        elif m := _TAB.fullmatch(path):
            self.tab_route(h, m, b)
        else:
            send_json(h, {"error": {"message": "not found"}}, HTTPStatus.NOT_FOUND)

    def tab_route(self, h, m: re.Match, b: dict) -> None:
        tid, tail, idx, remove = m.group(1), m.group(2), m.group(3), m.group(4)
        if tail is None:
            _fields(b, set(), {"name", "preamble", "color", "include"})
            if not b:
                raise Refused("nothing to change: give name, preamble, color or include")
            kw = {}
            if "name" in b:
                kw["name"] = _check_name(b["name"])
            if "preamble" in b:
                kw["preamble"] = "" if b["preamble"] == "" else _check_text(b["preamble"], "preamble")
            if "color" in b:
                kw["color"] = _color(b["color"])
            if "include" in b:
                if not isinstance(b["include"], bool):
                    raise Refused("include must be true or false")
                kw["include"] = b["include"]
            self.run_job(h, "tab_update", tid=tid, **kw)
        elif tail == "/remove":
            _fields(b, set())
            self.run_job(h, "tab_remove", tid=tid)
        elif tail == "/ask":
            _fields(b, {"text"})
            text = _check_text(b["text"], "text", MAX_ACTION)
            tab = self.tab(tid)  # an unknown tab is a 404 now, not an error event later
            if len(tab["messages"]) + 2 > MAX_MESSAGES:
                raise Refused(f"tab {tab['name']!r} is full ({MAX_MESSAGES} messages)")
            self.ask_route(tab)  # a tab whose reply can't be continued is a 400 now
            job = self.submit("ask", tid=tid, text=text)
            if job.error is not None:
                raise job.error
            send_json(h, {"queued": "ask"}, HTTPStatus.ACCEPTED)
        elif idx is None:
            _fields(b, {"role", "content"})
            if b["role"] not in ("user", "assistant"):
                raise Refused("role must be user or assistant")
            self.run_job(h, "message_add", tid=tid, role=b["role"], content=_check_text(b["content"], "content"))
        elif remove:
            _fields(b, set())
            self.run_job(h, "message_remove", tid=tid, index=_msg_index(idx))
        else:
            _fields(b, {"content"})
            self.run_job(h, "message_edit", tid=tid, index=_msg_index(idx), content=_check_text(b["content"], "content"))


_TAB = re.compile(r"/tabs/([^/]+)(/remove|/ask|/messages(?:/([^/]+)(/remove)?)?)?")
_REPLAY = re.compile(r"/decisions/([^/]+)/replay")


def _color(v) -> str:
    if not isinstance(v, str) or not _COLOR.fullmatch(v):
        raise Refused("color must be #rrggbb")
    return v.lower()


def _msg_index(s: str) -> int:
    if not (s.isascii() and s.isdigit()) or len(s) > 6:  # "²".isdigit() is True
        raise Refused(f"message index must be a number, got {s!r}")
    return int(s)


def send_json(h, obj, status=HTTPStatus.OK) -> None:
    body = json.dumps(obj).encode()
    h.send_response(status)
    h.send_header("content-type", "application/json")
    h.send_header("content-length", str(len(body)))
    h.send_header("cache-control", "no-store")
    h.end_headers()
    h.wfile.write(body)


def read_body(h) -> dict:
    cl = (h.headers.get("content-length") or "0").strip()
    if not (cl.isascii() and cl.isdigit()):
        h.close_connection = True
        raise Refused(f"Content-Length must be a non-negative integer, got {cl[:20]!r}")
    n = int(cl)
    if n > MAX_BODY:
        h.close_connection = True  # the body stays unread
        raise Refused(f"the body is too large ({n} bytes, at most {MAX_BODY})")
    raw = h.rfile.read(n) if n else b"{}"
    try:
        b = json.loads(raw or b"{}")
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise Refused(f"the body is not JSON: {e}") from None
    if not isinstance(b, dict):
        raise Refused("the body must be a JSON object")
    return b

