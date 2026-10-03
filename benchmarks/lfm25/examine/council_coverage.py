#!/usr/bin/env python3
"""Council option B: can a context say it has nothing to say?

Pre-registered in benchmarks/lfm25/council/coverage-v1.json (the scenario, its gold coverage cells, the
metrics and the success bar), committed before any read. Runs against a live lfm2d:

  1. uploads council-covers-v1 (a coverage question, then the verdict) and council-verdict-only-v1 (the same
     wording without it), and pins one held context per tab (POST /v1/contexts);
  2. per action, one multi-context read per spec (POST /v1/opinion with `contexts`), with `rendered: true` so every
     read's rendered prompt is hashed into the results;
  3. scores: per context and pooled, the AUC of P(covers = yes) for covered vs uncovered crisp cells, the means and
     counts on each side, the coverage slot's raw mass; then, secondary, pooled verdicts (linear; uniform vs
     weights = P(yes)) against each action's `rules` hint.

Field names come from the spec files (`required`: the coverage field first, the verdict last), never literals.
The scorer is pure (score()) and unit-tested in test_council_coverage.py.

  python3 benchmarks/lfm25/examine/council_coverage.py --base http://127.0.0.1:8095 \
      --out benchmarks/lfm25/results/2026-10-03-council-coverage.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
COUNCIL = HERE.parent / "council"
SCENARIO = COUNCIL / "coverage-v1.json"
COVERS = COUNCIL / "council-covers-v1.json"
VERDICT_ONLY = COUNCIL / "council-verdict-only-v1.json"


def call(base: str, method: str, path: str, body=None, raw: bytes | None = None, timeout: float = 300.0):
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    req = urllib.request.Request(base + path, data=data, method=method,
                                 headers={"content-type": "application/json"} if data else {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read() or b"null")
    except urllib.error.HTTPError as e:
        sys.exit(f"{method} {path}: {e.code} {e.read().decode(errors='replace')}")


def auc(pos: list[float], neg: list[float]) -> float | None:
    """P(a covered cell scores above an uncovered one), ties half (Mann-Whitney)."""
    if not pos or not neg:
        return None
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else 0.5 if p == n else 0.0
    return wins / (len(pos) * len(neg))


def pool_linear(probs: list[list[float]], weights: list[float]) -> list[float] | None:
    """Linear pool in request order (the daemon's `linear`); None when the weights sum to 0."""
    total = 0.0
    for w in weights:
        total += w
    if not total > 0:
        return None
    acc = [0.0] * len(probs[0])
    for p, w in zip(probs, weights):
        for i, x in enumerate(p):
            acc[i] += (w / total) * x
    s = sum(acc)
    return [x / s for x in acc]


def argmax(p: list[float]) -> int:
    best = 0
    for i in range(1, len(p)):
        if p[i] > p[best]:
            best = i
    return best


def score(rows: list[dict], tabs: list[str], yes: str, verdict_options: list[str]) -> dict:
    """Aggregates over the per-cell rows `run` records.

    Each row: {action, rules, coverage: {tab: yes|no|ambiguous}, cells: {tab: {p_yes, covers_mass, verdict: [p...]}},
    baseline: {tab: [p...]}}.
    """
    out: dict = {"per_context": {}, "pooled": {}}
    all_pos, all_neg = [], []
    for tab in tabs:
        pos = [r["cells"][tab]["p_yes"] for r in rows if r["coverage"][tab] == "yes"]
        neg = [r["cells"][tab]["p_yes"] for r in rows if r["coverage"][tab] == "no"]
        amb = [r["cells"][tab]["p_yes"] for r in rows if r["coverage"][tab] == "ambiguous"]
        masses = [r["cells"][tab]["covers_mass"] for r in rows]
        all_pos += pos
        all_neg += neg
        out["per_context"][tab] = {
            "covered": len(pos), "uncovered": len(neg), "ambiguous": len(amb),
            "auc": auc(pos, neg),
            "mean_p_yes_covered": sum(pos) / len(pos) if pos else None,
            "mean_p_yes_uncovered": sum(neg) / len(neg) if neg else None,
            "covered_at_or_above_half": sum(p >= 0.5 for p in pos),
            "uncovered_below_half": sum(p < 0.5 for p in neg),
            "ambiguous_p_yes": amb,
            "covers_mass_min": min(masses), "covers_mass_median": sorted(masses)[len(masses) // 2],
        }
    out["pooled"] = {"auc": auc(all_pos, all_neg), "covered": len(all_pos), "uncovered": len(all_neg)}
    per = out["per_context"]
    out["success"] = all(
        p["auc"] is not None and p["auc"] >= 0.80 and p["uncovered_below_half"] * 3 >= p["uncovered"] * 2
        for p in per.values()
    )
    hits = {"baseline_uniform": 0, "covers_uniform": 0, "covers_weighted": 0}
    weighted_none = 0
    verdicts = []
    for r in rows:
        base = pool_linear([r["baseline"][t] for t in tabs], [1.0] * len(tabs))
        uni = pool_linear([r["cells"][t]["verdict"] for t in tabs], [1.0] * len(tabs))
        wtd = pool_linear([r["cells"][t]["verdict"] for t in tabs], [r["cells"][t]["p_yes"] for t in tabs])
        pick = {k: (verdict_options[argmax(p)] if p else None)
                for k, p in (("baseline_uniform", base), ("covers_uniform", uni), ("covers_weighted", wtd))}
        weighted_none += wtd is None
        for k, v in pick.items():
            hits[k] += v == r["rules"]
        verdicts.append({"action": r["action"], "rules": r["rules"], **pick})
    out["secondary"] = {"against_rules_hint": hits, "of": len(rows), "weighted_unpoolable": weighted_none,
                        "verdicts": verdicts}
    return out


def run(base: str) -> dict:
    scenario = json.loads(SCENARIO.read_text())
    covers_raw, only_raw = COVERS.read_bytes(), VERDICT_ONLY.read_bytes()
    covers_spec, only_spec = json.loads(covers_raw), json.loads(only_raw)
    covers_field, verdict_field = covers_spec["output_schema"]["required"][0], covers_spec["output_schema"]["required"][-1]
    covers_options = covers_spec["output_schema"]["properties"][covers_field]["enum"]
    yes = covers_options[0]
    verdict_options = covers_spec["output_schema"]["properties"][verdict_field]["enum"]
    assert only_spec["output_schema"]["properties"][verdict_field]["enum"] == verdict_options
    covers_id = call(base, "POST", "/v1/opinion/specs", raw=covers_raw)["id"]
    only_id = call(base, "POST", "/v1/opinion/specs", raw=only_raw)["id"]
    tabs = [t["name"] for t in scenario["tabs"]]
    ids, lengths = [], {}
    for t in scenario["tabs"]:
        built = call(base, "POST", "/v1/contexts", {
            "system": f"{scenario['reviewer']}\n\nTHIS SOURCE: {t['name']}\n{t['preamble']}",
            "messages": [{"role": "user", "content": m} for m in t["messages"]],
            "pin": True,
        })
        ids.append(built["id"])
        lengths[t["name"]] = built["n_tokens"]
    rows, rendered, timings = [], [], []
    for a in scenario["actions"]:
        t0 = time.perf_counter()
        r = call(base, "POST", "/v1/opinion", {
            "spec": covers_id, "state": {"input": a["text"]}, "contexts": ids, "rendered": True,
            "questions": [{"field": covers_field}, {"field": verdict_field}], "timeout_ms": 120000,
        })
        t1 = time.perf_counter()
        b = call(base, "POST", "/v1/opinion", {
            "spec": only_id, "state": {"input": a["text"]}, "contexts": ids, "rendered": True,
            "questions": [{"field": verdict_field}], "timeout_ms": 120000,
        })
        timings.append({"covers_ms": (t1 - t0) * 1000, "baseline_ms": (time.perf_counter() - t1) * 1000})
        cells, baseline = {}, {}
        for tab, read, bread in zip(tabs, r["reads"], b["reads"]):
            ans = {x["field"]: x for x in read["answers"]}
            cov, ver = ans[covers_field], ans[verdict_field]
            cells[tab] = {
                "p_yes": next(o["prob"] for o in cov["options"] if o["option"] == yes),
                "covers_mass": math.exp(cov["sequence_mass"]),
                "verdict": [o["prob"] for o in ver["options"]],
                "verdict_mass": math.exp(ver["sequence_mass"]),
                "context_tokens": read.get("context_tokens"),
            }
            bans = {x["field"]: x for x in bread["answers"]}[verdict_field]
            baseline[tab] = [o["prob"] for o in bans["options"]]
            for x in (read, bread):
                rendered.append(hashlib.sha256(x["rendered"].encode()).hexdigest())
        rows.append({"action": a["text"], "rules": a["rules"], "coverage": a["coverage"], "cells": cells,
                     "baseline": baseline, "baseline_mass": {
                         tab: math.exp({x["field"]: x for x in br["answers"]}[verdict_field]["sequence_mass"])
                         for tab, br in zip(tabs, b["reads"])}})
        print(f"{a['text'][:48]:48}  " + "  ".join(f"{t[0]}:{cells[t]['p_yes']:.2f}/{a['coverage'][t][0]}" for t in tabs),
              flush=True)
    for cid in ids:
        call(base, "DELETE", f"/v1/contexts/{cid}")
    info = call(base, "GET", "/v1/adjudicator")
    return {
        "date": time.strftime("%Y-%m-%d"),
        "daemon": {k: info.get(k) for k in ("model_id", "weight_hash", "backend", "device", "candle_rev", "dtype")},
        "scenario_sha256": hashlib.sha256(SCENARIO.read_bytes()).hexdigest(),
        "covers_spec_id": covers_id, "verdict_only_spec_id": only_id,
        "rendered_sha256": hashlib.sha256("".join(rendered).encode()).hexdigest(),
        "context_tokens": lengths,
        "fields": {"covers": covers_field, "yes": yes, "verdict": verdict_field, "verdict_options": verdict_options},
        "timing_ms": {"covers_median": sorted(t["covers_ms"] for t in timings)[len(timings) // 2],
                      "baseline_median": sorted(t["baseline_ms"] for t in timings)[len(timings) // 2]},
        "score": score(rows, tabs, yes, verdict_options),
        "rows": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base", default="http://127.0.0.1:8095")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    result = run(args.base)
    Path(args.out).write_text(json.dumps(result, indent=1) + "\n")
    s = result["score"]
    print(json.dumps({"per_context": {t: {k: v for k, v in p.items() if k != "ambiguous_p_yes"}
                                      for t, p in s["per_context"].items()},
                      "pooled": s["pooled"], "success": s["success"],
                      "secondary": {k: v for k, v in s["secondary"].items() if k != "verdicts"}}, indent=1))


if __name__ == "__main__":
    main()
