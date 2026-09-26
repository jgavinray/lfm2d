#!/usr/bin/env python3
"""Measure a /tail scenario's spec on its own feed, through the daemon.

    python3 demo/web/tail_eval.py --url http://127.0.0.1:18191 --scenario 'Trip to Lisbon'

For each scenario: upload its spec's exact bytes, check the judged field and
its keep option against the menu, start a chat with the scenario's system
prompt and opening instruction (the daemon's own generation, not streamed),
then read every item of the feed three ways: at the chat's `checkpoint_user`
(where the page reads while the chat is still reasoning), at `checkpoint`
(after the assistant's turn), and without the chat (the spec's own prompt,
which never saw what the user asked for); `--follow-up` sends the scenario's
follow-up as turn 2, reads again after it, scores against the author's answer
after the follow-up and counts how many flipped items P(keep) followed. Prints, per position, how the
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


def moves(before, after, fits_before, fits_after):
    """How P(keep) moved between two reads of the same items when the author's
    answer changed (a flip) and when it did not. A flip is followed when P
    moved the way the answer did."""
    flips = [(b, a, fa) for b, a, fb, fa in zip(before, after, fits_before, fits_after) if fb != fa]
    up = [(a > b) for b, a, fa in flips if fa]
    down = [(a < b) for b, a, fa in flips if not fa]
    steady = [abs(a - b) for b, a, fb, fa in zip(before, after, fits_before, fits_after) if fb == fa]
    return {"flips": len(flips), "followed": sum(up) + sum(down),
            "to_fits": [sum(up), len(up)], "to_breaks": [sum(down), len(down)],
            "steady_abs_move_p50": round(statistics.median(steady), 3) if steady else None}


def read_feed(url, sc, menu, field, keep, info, label, checkpoint, gold, verbose):
    """Read every item at one checkpoint (None: no chat), print the summary
    against the gold key, return the P(keep) of each item."""
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
        fits = item[gold]
        rows.append({"fits": fits, "chat_only": item["chat_only"], "p_keep": p[keep],
                     "mass": math.exp(a["sequence_mass"]), "ms": ms, "prefix": r["cache"]["prefix"]})
        if verbose:
            flip = " FLIP" if item["fits"] != item["fits_after_follow_up"] and gold != "fits" else ""
            print(f"  {'fits  ' if fits else 'breaks'} P({keep})={p[keep]:.3f} mass={rows[-1]['mass']:.4f} "
                  f"{ms:5.0f} ms {r['cache']['prefix']:10} {item['from']}: {item['title']}{flip}")
    keep_at, drop_below = sc["filter"]["keep_at"], sc["filter"]["drop_below"]
    s = summarize(rows, keep_at=keep_at, drop_below=drop_below, min_mass=MIN_MASS)
    auc_s = "n/a" if s["auc"] is None else f"{s['auc']:.3f}"
    print(f"top option {keep}: fits {s['fits']['top_keep']}/{s['fits']['n']}, breaks "
          f"{s['breaks']['top_keep']}/{s['breaks']['n']}\nbins at keep>={keep_at}, drop<{drop_below}: "
          f"fits {s['fits']}, breaks {s['breaks']}\nAUC P({keep}) fits vs breaks {auc_s} · raw mass min "
          f"{s['mass_min']:.4f} p50 {s['mass_p50']:.4f} · read p50 {s['ms_p50']:.0f} ms · prefix "
          f"{dict((k, sum(r['prefix'] == k for r in rows)) for k in sorted({r['prefix'] for r in rows}))}")
    return [r["p_keep"] for r in rows]


def chat_turn(url, body):
    t0 = time.perf_counter()
    r = call(url, "/v1/chat", body)
    r["wall_ms"] = 1000 * (time.perf_counter() - t0)
    print(f"chat turn: {r['prompt_tokens']} prompt tokens ({r['cached_tokens']} resident), "
          f"{r['completion_tokens']} generated ({r['finish_reason']}) in {r['wall_ms'] / 1000:.1f} s, "
          f"checkpoint_user {r['checkpoint_user'][:10]}, checkpoint {str(r['checkpoint'])[:10]}")
    return r


def measure(url, sc, verbose, spec_path=None, positions=("user", "assistant", "none"), chat_tokens=2048,
            judge=None, follow_up=False):
    spec_path = Path(spec_path) if spec_path else STATIC / sc["spec"].lstrip("/")
    spec_bytes = spec_path.read_bytes()
    reg = call(url, "/v1/opinion/specs", raw=spec_bytes)
    menu = next(m for m in call(url, "/v1/opinion/specs") if m["id"] == reg["id"])
    field, keep = judge or (sc["judge"]["field"], sc["judge"]["keep"])
    info = next((f for f in menu["fields"] if f["field"] == field), None)
    if info is None or info["kind"] != "choice" or keep not in info["options"]:
        sys.exit(f"{sc['name']}: the menu has no choice field {field!r} with option {keep!r}")
    model = call(url, "/v1/adjudicator")
    print(f"## {sc['name']}\nspec {spec_path.name} id {menu['id'][:16]} · {model['model_id']} {model['device']} "
          f"candle {model['candle_rev'][:12]}")
    turns = [chat_turn(url, {"system": sc["system"], "messages": [{"role": "user", "content": sc["first"]}],
                             "max_tokens": chat_tokens, "timeout_ms": 600000})]
    if follow_up:
        if turns[0]["checkpoint"] is None:
            sys.exit("turn 1 hit max_tokens: no checkpoint to send the follow-up from")
        turns.append(chat_turn(url, {"from": turns[0]["checkpoint"], "timeout_ms": 600000, "max_tokens": chat_tokens,
                                     "messages": [{"role": "user", "content": sc["follow_up"]}]}))
    read = {}
    for n, chat in enumerate(turns, 1):
        gold = "fits" if n == 1 else "fits_after_follow_up"
        at = {"user": (f"turn {n}: after the user's message (mid-turn)", chat["checkpoint_user"]),
              "assistant": (f"turn {n}: after the assistant's turn", chat["checkpoint"])}
        for pos in positions:
            if pos == "none":
                continue
            label, checkpoint = at[pos]
            if checkpoint is None:
                print(f"\n### {label}: no checkpoint (the turn hit max_tokens)")
                continue
            read[(n, pos)] = read_feed(url, sc, menu, field, keep, info, label, checkpoint, gold, verbose)
    if "none" in positions:
        read_feed(url, sc, menu, field, keep, info, "without the chat", None, "fits", verbose)
    if follow_up:
        fb = [i["fits"] for i in sc["items"]]
        fa = [i["fits_after_follow_up"] for i in sc["items"]]
        for pos in positions:
            if (1, pos) in read and (2, pos) in read:
                print(f"\nturn 1 -> turn 2, {pos}: {moves(read[(1, pos)], read[(2, pos)], fb, fa)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", required=True)
    ap.add_argument("--scenario", action="append", help="scenario name (repeatable; default all)")
    ap.add_argument("--spec", help="measure this spec file instead of the scenario's own (one scenario)")
    ap.add_argument("--judge", help="FIELD=OPTION to filter on, with --spec (default: the scenario's)")
    ap.add_argument("--at", default="user,assistant,none",
                    help="read positions: user (mid-turn), assistant (after the turn), none (no chat)")
    ap.add_argument("--chat-tokens", type=int, default=2048, help="max_tokens for the chat turn")
    ap.add_argument("--follow-up", action="store_true",
                    help="send the scenario's follow-up as turn 2 and score the items that flip")
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
        measure(a.url.rstrip("/"), sc, a.verbose, a.spec, positions, a.chat_tokens, judge, a.follow_up)


if __name__ == "__main__":
    main()
