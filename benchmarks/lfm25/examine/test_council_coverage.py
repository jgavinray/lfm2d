"""The coverage scorer (council_coverage.score) on hand-built rows: it must be able to fail."""
import json
from pathlib import Path

import council_coverage as cc

TABS = ["A", "B"]
OPTS = ["allow", "ask", "report"]


def row(rules, cov, p_yes, verdicts, baseline=None):
    return {
        "action": f"act-{rules}-{p_yes}", "rules": rules, "coverage": dict(zip(TABS, cov)),
        "cells": {t: {"p_yes": p, "covers_mass": 0.99, "verdict": v} for t, p, v in zip(TABS, p_yes, verdicts)},
        "baseline": {t: v for t, v in zip(TABS, baseline or verdicts)},
    }


def test_auc_is_mann_whitney_with_half_ties():
    assert cc.auc([0.9, 0.8], [0.1, 0.2]) == 1.0
    assert cc.auc([0.1], [0.9]) == 0.0
    assert cc.auc([0.5], [0.5]) == 0.5
    assert cc.auc([], [0.1]) is None


def test_a_clean_split_succeeds_and_a_hedging_context_fails():
    clean = [
        row("allow", ["yes", "no"], [0.9, 0.1], [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]]),
        row("ask", ["no", "yes"], [0.2, 0.95], [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]]),
        row("ask", ["no", "no"], [0.1, 0.05], [[0.3, 0.6, 0.1], [0.3, 0.6, 0.1]]),
    ]
    s = cc.score(clean, TABS, "yes", OPTS)
    assert s["per_context"]["A"]["auc"] == 1.0 and s["success"] is True
    # B says yes to everything: AUC can still be fine on two cells, but every uncovered cell is >= 0.5.
    hedge = [dict(r, cells={**r["cells"], "B": {**r["cells"]["B"], "p_yes": 0.97}}) for r in clean]
    s = cc.score(hedge, TABS, "yes", OPTS)
    assert s["per_context"]["B"]["uncovered_below_half"] == 0 and s["success"] is False


def test_ambiguous_cells_stay_out_of_the_primary_metric():
    rows = [
        row("allow", ["yes", "ambiguous"], [0.9, 0.0], [[1, 0, 0], [1, 0, 0]]),
        row("allow", ["no", "ambiguous"], [0.1, 1.0], [[1, 0, 0], [1, 0, 0]]),
    ]
    s = cc.score(rows, TABS, "yes", OPTS)
    b = s["per_context"]["B"]
    assert (b["covered"], b["uncovered"], b["ambiguous"], b["auc"]) == (0, 0, 2, None)
    assert b["ambiguous_p_yes"] == [0.0, 1.0]
    assert s["success"] is False, "a context with no crisp cells cannot pass"


def test_coverage_weights_move_the_pooled_verdict():
    # A covers it and says ask; B does not and says allow louder. Uniform picks allow, weighted picks ask.
    r = row("ask", ["yes", "no"], [0.9, 0.05], [[0.2, 0.7, 0.1], [0.95, 0.04, 0.01]])
    s = cc.score([r], TABS, "yes", OPTS)["secondary"]
    assert s["verdicts"][0]["covers_uniform"] == "allow"
    assert s["verdicts"][0]["covers_weighted"] == "ask"
    assert s["against_rules_hint"] == {"baseline_uniform": 0, "covers_uniform": 0, "covers_weighted": 1}


def test_the_scenario_is_well_formed():
    s = json.loads((Path(cc.SCENARIO)).read_text())
    tabs = [t["name"] for t in s["tabs"]]
    for a in s["actions"]:
        assert set(a["coverage"]) == set(tabs), a["text"]
        assert set(a["coverage"].values()) <= {"yes", "no", "ambiguous"}, a["text"]
        assert a["rules"] in OPTS
    spec = json.loads(Path(cc.COVERS).read_text())
    assert spec["output_schema"]["required"][-1] == "verdict"
    assert spec["output_schema"]["properties"]["verdict"]["enum"] == OPTS
