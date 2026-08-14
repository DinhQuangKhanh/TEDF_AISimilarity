#!/usr/bin/env python3
"""Standalone checker: does the code reproduce the DASSF paper's numbers?

Reads tests/fixtures/paper_formulas.json, runs each case through the real code,
and prints a PASS/FAIL report. Run from the project root (inside the app image or
a venv that has the app dependencies):

    python tools/verify_formulas.py

Exit code is non-zero if any case fails, so it doubles as a CI check.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.ontology.sedo import get_sedo  # noqa: E402
from app.utils.score_calculator import (  # noqa: E402
    WEIGHTS,
    action_for,
    composite_score,
    is_structural_duplication,
    level_for,
)

DATA = json.loads((ROOT / "tests" / "fixtures" / "paper_formulas.json").read_text(encoding="utf-8"))
sedo = get_sedo()

passed = 0
failed = 0


def check(name, got, expected, tol=0.0, source=""):
    global passed, failed
    ok = (abs(got - expected) <= tol) if isinstance(expected, (int, float)) and not isinstance(expected, bool) else (got == expected)
    status = "PASS" if ok else "FAIL"
    passed += ok
    failed += not ok
    print(f"[{status}] {name}: got={got} expected={expected}  ({source})")


for c in DATA["wu_palmer"]:
    check(f"wu_palmer({c['c1']},{c['c2']})", round(sedo.wu_palmer(c["c1"], c["c2"]), 4), c["expected"], c["tol"], c["source"])

for c in DATA["wpath"]:
    check(f"wpath({c['c1']},{c['c2']})", round(sedo.wpath(c["c1"], c["c2"]), 4), c["expected"], c["tol"], c["source"])

w = DATA["mddm_weights"]
check("mddm_weights", WEIGHTS, {k: w[k] for k in ("semantic", "lexical", "structure", "domain")}, source=w["source"])

for c in DATA["composite"]:
    check(
        f"composite({c['s_sem']},{c['s_lex']},{c['s_str']},{c['s_dom']})",
        composite_score(c["s_sem"], c["s_lex"], c["s_str"], c["s_dom"]),
        c["expected"],
        c["tol"],
        c["source"],
    )

for c in DATA["levels"]:
    lvl = level_for(c["score"])
    check(f"level_for({c['score']})", (lvl, action_for(lvl)), (c["level"], c["action"]), source=c["source"])

for c in DATA["structural_duplication"]:
    check(
        f"is_structural_duplication(S_str={c['s_str']},S_dom={c['s_dom']})",
        is_structural_duplication(c["s_str"], c["s_dom"]),
        c["expected"],
        source=c["source"],
    )

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
