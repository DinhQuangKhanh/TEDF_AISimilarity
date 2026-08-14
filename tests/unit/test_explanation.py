"""Tests for the Module-4 explanation report (app/services/explanation.py)."""

from app.services.explanation import build_explanation


class FakeThesis:
    def __init__(self, title):
        self.title = title
        self.description = ""
        self.scope = ""
        self.objectives = ""
        self.expected_result = ""


def _scores(sem, lex, struct, dom, overall, level):
    return {
        "semantic_score": sem, "lexical_score": lex, "structure_score": struct,
        "domain_score": dom, "overall_score": overall, "level": level,
    }


_HOTEL = FakeThesis("Hotel Management System with React and Node.js")
_PHARMACY = FakeThesis("Pharmacy Management System with React and Node.js")
_HOTEL2 = FakeThesis("Hotel booking platform with React and Node.js")


# ── structure of the report ─────────────────────────────────────────────────────────
def test_report_has_full_breakdown_and_action():
    rep = build_explanation(_HOTEL, _PHARMACY, _scores(0.4, 0.5, 0.98, 0.20, 0.55, "Moderate"))
    assert set(rep["breakdown"]) == {"semantic", "lexical", "structure", "domain"}
    assert rep["breakdown"]["structure"] == 0.98
    assert rep["level"] == "Moderate"
    assert rep["action"] == "Warn; committee reviews"
    assert rep["overall_score"] == 0.55


# ── structural duplication (paper's canonical case) ─────────────────────────────────
def test_structural_duplication_names_shared_stack_and_suggests_swap():
    rep = build_explanation(_HOTEL, _PHARMACY, _scores(0.4, 0.5, 0.98, 0.20, 0.55, "Moderate"))
    assert rep["is_structural_duplication"] is True
    assert set(rep["shared_concepts"]["tech"]) == {"React", "Node.js"}
    assert rep["shared_concepts"]["domain"] == []  # hotel vs pharmacy do NOT share a domain
    assert "same tech stack with a different business domain" in rep["reasons"]
    suggestion = rep["revision_suggestion"].lower()
    assert "architecture" in suggestion or "technical" in suggestion
    assert "React" in rep["revision_suggestion"] or "Node.js" in rep["revision_suggestion"]


def test_structural_reason_not_contradictory():
    rep = build_explanation(_HOTEL, _PHARMACY, _scores(0.4, 0.5, 0.98, 0.20, 0.55, "Moderate"))
    # a structural duplication must never also claim "same business domain"
    assert not any("same business domain" in r for r in rep["reasons"])


# ── both-domain overlap ─────────────────────────────────────────────────────────────
def test_both_domain_and_structure_suggests_rethink():
    rep = build_explanation(_HOTEL, _HOTEL2, _scores(0.8, 0.8, 0.9, 0.9, 0.86, "Critical"))
    assert rep["is_structural_duplication"] is False
    assert "Hotel" in rep["shared_concepts"]["domain"]
    assert "BOTH" in rep["revision_suggestion"]
    assert rep["action"] == "Reject"


# ── semantic/lexical closeness without a shared stack ───────────────────────────────
def test_wording_closeness_suggests_narrowing():
    a = FakeThesis("Smart tutoring platform")
    b = FakeThesis("Intelligent tutoring system")
    rep = build_explanation(a, b, _scores(0.82, 0.7, 0.3, 0.1, 0.5, "Moderate"))
    assert "narrow" in rep["revision_suggestion"].lower() or "clarify" in rep["revision_suggestion"].lower()


# ── same domain, different stack ────────────────────────────────────────────────────
def test_same_domain_different_stack_message():
    a = FakeThesis("Hotel management with React")
    b = FakeThesis("Hotel management with Spring Boot")
    rep = build_explanation(a, b, _scores(0.4, 0.3, 0.3, 0.7, 0.44, "Moderate"))
    assert "Hotel" in rep["shared_concepts"]["domain"]
    assert "domain" in rep["revision_suggestion"].lower()


# ── low similarity ──────────────────────────────────────────────────────────────────
def test_low_similarity_requires_no_revision():
    rep = build_explanation(_HOTEL, FakeThesis("Weather forecast API"),
                            _scores(0.1, 0.1, 0.1, 0.0, 0.08, "Low"))
    assert rep["revision_suggestion"] == "Sufficiently distinct from existing topics; no revision required."
    assert rep["action"] == "Accept"


# ── level derived when omitted ──────────────────────────────────────────────────────
def test_level_is_derived_when_absent():
    scores = {"semantic_score": 0.9, "lexical_score": 0.9, "structure_score": 1.0,
              "domain_score": 1.0, "overall_score": 0.95}
    rep = build_explanation(_HOTEL, _HOTEL2, scores)
    assert rep["level"] == "Critical"
