#!/usr/bin/env python3
"""Measure a /tail scenario's spec on its own feed, through the daemon.

    python3 demo/web/tail_eval.py --url http://127.0.0.1:18191 --scenario 'Trip to Lisbon'

For each scenario: upload its spec's exact bytes, check the judged field and
its keep option against the menu, start a chat with the scenario's system
prompt and opening instruction (the daemon's own generation, not streamed),
then read every item of the feed three ways: at the chat's `checkpoint_user`
(where the page reads while the chat is still reasoning), at `checkpoint`
(after the assistant's turn), and without the chat (the spec's own prompt,
which never saw what the user asked for). Prints, per position, how the
items the author says fit and break were split by the top option and by the
page's default bins, the AUC of P(keep) between them, and the raw mass on
the answer set; `-v` adds one line per item (the items are invented props).
The rendered prompt of the first read is hashed and grepped for the options'
words, which framing text can pump.
"""
import argparse
import hashlib
import json
import math
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

STATIC = Path(__file__).resolve().parent / "static"
# The page never keeps or drops a read whose answer set held under MIN_MASS
# of the raw probability (tail.html); its keep/drop thresholds are each
# scenario's `filter` defaults.
MIN_MASS = 0.5


def auc(pos, neg):
    """P(a positive scores above a negative), ties half."""
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def bin_of(p_keep, mass, *, keep_at, drop_below, min_mass):
    if mass < min_mass:
        return "maybe"
    if p_keep >= keep_at:
        return "keep"
    if p_keep < drop_below:
        return "drop"
    return "maybe"


def summarize(rows, *, keep_at, drop_below, min_mass):
    def split(group):
        out = {"n": len(group), "top_keep": sum(r["p_keep"] > 0.5 for r in group),
               "keep": 0, "maybe": 0, "drop": 0}
        for r in group:
            out[bin_of(r["p_keep"], r["mass"], keep_at=keep_at, drop_below=drop_below, min_mass=min_mass)] += 1
        return out
    fits = [r for r in rows if r["fits"]]
    breaks = [r for r in rows if not r["fits"]]
    masses = [r["mass"] for r in rows]
    return {
        "fits": split(fits), "breaks": split(breaks),
        "chat_only": split([r for r in rows if r["chat_only"]]),
        "auc": auc([r["p_keep"] for r in fits], [r["p_keep"] for r in breaks]),
        "mass_min": min(masses), "mass_p50": statistics.median(masses),
        "ms_p50": statistics.median(r["ms"] for r in rows),
    }


def call(url, path, body=None, raw=None, timeout=600):
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    req = urllib.request.Request(url + path, data=data, headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        sys.exit(f"{path}: {e.code} {e.read().decode(errors='replace')}")


def measure(url, sc, verbose, spec_path=None, positions=("user", "assistant", "none"), chat_tokens=2048, judge=None):
    spec_path = Path(spec_path) if spec_path else STATIC / sc["spec"].lstrip("/")
    spec_bytes = spec_path.read_bytes()
    reg = call(url, "/v1/opinion/specs", raw=spec_bytes)
    menu = next(m for m in call(url, "/v1/opinion/specs") if m["id"] == reg["id"])
    field, keep = judge or (sc["judge"]["field"], sc["judge"]["keep"])
    info = next((f for f in menu["fields"] if f["field"] == field), None)
    if info is None or info["kind"] != "choice" or keep not in info["options"]:
        sys.exit(f"{sc['name']}: the menu has no choice field {field!r} with option {keep!r}")
    model = call(url, "/v1/adjudicator")
    chat = call(url, "/v1/chat", {"system": sc["system"], "messages": [{"role": "user", "content": sc["first"]}],
                                  "max_tokens": chat_tokens, "timeout_ms": 600000})
    print(f"## {sc['name']}\nspec {spec_path.name} id {menu['id'][:16]} · {model['model_id']} {model['device']} "
          f"candle {model['candle_rev'][:12]}\nchat: {chat['prompt_tokens']} prompt tokens, "
          f"{chat['completion_tokens']} generated ({chat['finish_reason']}), checkpoint_user "
          f"{chat['checkpoint_user'][:10]}, checkpoint {str(chat['checkpoint'])[:10]}")
    at = {"user": ("after the user's message (mid-turn)", chat["checkpoint_user"]),
          "assistant": ("after the assistant's turn", chat["checkpoint"]), "none": ("without the chat", None)}
    for label, checkpoint in (at[p] for p in positions):
        if label.startswith("after the assistant") and checkpoint is None:
            print(f"\n### {label}: no checkpoint (the turn hit max_tokens)")
            continue
        rows = []
        for n, item in enumerate(sc["items"]):
            body = {"spec": menu["id"], "state": {"input": item["text"]}, "questions": [{"field": field}],
                    "use_cache": True, "rendered": n == 0}
            if checkpoint:
                body["context"] = {"checkpoint": checkpoint}
            t0 = time.perf_counter()
            r = call(url, "/v1/opinion", body)
            ms = 1000 * (time.perf_counter() - t0)
            a = r["answers"][0]
            p = {o["option"]: o["prob"] for o in a["options"]}
            if n == 0:
                text = r["rendered"]
                sha = hashlib.sha256(text.encode()).hexdigest()
                if sha != a["rendered_sha256"]:
                    sys.exit("rendered text does not hash to rendered_sha256")
                words = {o: len(re.findall(rf"\b{re.escape(o)}", text, re.I)) for o in info["options"]}
                print(f"\n### {label}\nrendered sha256 {sha[:16]} ({r['prompt_tokens']} prompt tokens); "
                      f"option words in the rendered prompt: {words}")
            rows.append({"fits": item["fits"], "chat_only": item["chat_only"], "p_keep": p[keep],
                         "mass": math.exp(a["sequence_mass"]), "ms": ms,
                         "described": [d["value"] for d in r["described"]]})
            if verbose:
                tag = "fits  " if item["fits"] else ("chat! " if item["chat_only"] else "breaks")
                print(f"  {tag} P({keep})={p[keep]:.3f} mass={rows[-1]['mass']:.4f} {ms:5.0f} ms  "
                      f"{item['from']}: {item['note']}")
        keep_at, drop_below = sc["filter"]["keep_at"], sc["filter"]["drop_below"]
        s = summarize(rows, keep_at=keep_at, drop_below=drop_below, min_mass=MIN_MASS)
        auc_s = "n/a" if s["auc"] is None else f"{s['auc']:.3f}"
        print(f"top option {keep}: fits {s['fits']['top_keep']}/{s['fits']['n']}, breaks "
              f"{s['breaks']['top_keep']}/{s['breaks']['n']} (chat-only {s['chat_only']['top_keep']}/"
              f"{s['chat_only']['n']})\nbins at keep>={keep_at}, drop<{drop_below}: fits {s['fits']}, "
              f"breaks {s['breaks']}\nAUC P({keep}) fits vs breaks {auc_s} · raw mass min "
              f"{s['mass_min']:.4f} p50 {s['mass_p50']:.4f} · read p50 {s['ms_p50']:.0f} ms")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", required=True)
    ap.add_argument("--scenario", action="append", help="scenario name (repeatable; default all)")
    ap.add_argument("--spec", help="measure this spec file instead of the scenario's own (one scenario)")
    ap.add_argument("--judge", help="FIELD=OPTION to filter on, with --spec (default: the scenario's)")
    ap.add_argument("--at", default="user,assistant,none",
                    help="read positions: user (mid-turn), assistant (after the turn), none (no chat)")
    ap.add_argument("--chat-tokens", type=int, default=2048, help="max_tokens for the chat turn")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    scenarios = json.loads((STATIC / "tail.json").read_text())["scenarios"]
    chosen = [s for s in scenarios if not a.scenario or s["name"] in a.scenario]
    if a.scenario and len(chosen) != len(a.scenario):
        sys.exit(f"unknown scenario; have {[s['name'] for s in scenarios]}")
    if a.spec and len(chosen) != 1:
        sys.exit("--spec needs exactly one --scenario")
    positions = a.at.split(",")
    if not positions or set(positions) - {"user", "assistant", "none"}:
        sys.exit("--at takes user, assistant and none")
    for sc in chosen:
        judge = tuple(a.judge.split("=", 1)) if a.judge else None
        measure(a.url.rstrip("/"), sc, a.verbose, a.spec, positions, a.chat_tokens, judge)


if __name__ == "__main__":
    main()
