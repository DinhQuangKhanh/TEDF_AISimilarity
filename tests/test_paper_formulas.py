"""Verify that the implementation reproduces the numeric claims of the DASSF paper.

Each case comes from tests/fixtures/paper_formulas.json (input + expected value +
the paper location it is taken from). Run with:

    pytest tests/test_paper_formulas.py -v
"""

import json
from pathlib import Path

import pytest

from app.core.config import _PAPER_WEIGHTS
from app.ontology.sedo import get_sedo
from app.utils.score_calculator import (
    action_for,
    composite_score,
    is_structural_duplication,
    level_for,
)

_FIXTURE = Path(__file__).parent / "fixtures" / "paper_formulas.json"
DATA = json.loads(_FIXTURE.read_text(encoding="utf-8"))


def _id(case: dict) -> str:
    return case.get("source", "case")[:70]


@pytest.mark.parametrize("case", DATA["wu_palmer"], ids=_id)
def test_wu_palmer(case):
    value = get_sedo().wu_palmer(case["c1"], case["c2"])
    assert value == pytest.approx(case["expected"], abs=case["tol"])


@pytest.mark.parametrize("case", DATA["wpath"], ids=_id)
def test_wpath(case):
    value = get_sedo().wpath(case["c1"], case["c2"])
    assert value == pytest.approx(case["expected"], abs=case["tol"])


def test_mddm_weights():
    # The paper's Eq.1 weights are the code's PAPER default. (The deployed weights may be tuned/env
    # overrides — that is a deployment choice, not a claim about the paper.)
    w = DATA["mddm_weights"]
    assert _PAPER_WEIGHTS == {"semantic": w["semantic"], "lexical": w["lexical"],
                             "structure": w["structure"], "domain": w["domain"]}
    assert round(sum(_PAPER_WEIGHTS.values()), 6) == 1.0


@pytest.mark.parametrize("case", DATA["composite"], ids=_id)
def test_composite(case):
    # The paper's worked examples use the paper weights; pass them explicitly so this test verifies
    # the FORMULA regardless of the deployed weight vector.
    value = composite_score(case["s_sem"], case["s_lex"], case["s_str"], case["s_dom"], _PAPER_WEIGHTS)
    assert value == pytest.approx(case["expected"], abs=case["tol"])


@pytest.mark.parametrize("case", DATA["levels"], ids=_id)
def test_levels(case):
    level = level_for(case["score"])
    assert level == case["level"]
    assert action_for(level) == case["action"]


@pytest.mark.parametrize("case", DATA["structural_duplication"], ids=_id)
def test_structural_duplication(case):
    assert is_structural_duplication(case["s_str"], case["s_dom"]) is case["expected"]
