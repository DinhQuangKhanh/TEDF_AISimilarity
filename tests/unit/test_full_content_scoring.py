"""Tests for P4: full-content comparison (semantic/lexical) + recent-semester filtering."""

from app.services.lexical import build_lexical_scorer
from app.utils import score_calculator as sc


class Topic:
    def __init__(self, title="", description="", scope="", objectives="", expected_result="", semester=None):
        self.title = title
        self.description = description
        self.scope = scope
        self.objectives = objectives
        self.expected_result = expected_result
        self.semester = semester
        self.technologies = []
        self.structures = []
        self.domains = []


# ── _has_body ───────────────────────────────────────────────────────────────────────
def test_has_body_true_when_any_field_present():
    assert sc._has_body(Topic(title="X", description="some detail")) is True
    assert sc._has_body(Topic(title="X", scope="a scope")) is True


def test_has_body_false_for_title_only():
    assert sc._has_body(Topic(title="Just a title")) is False


# ── _pair_texts (full when both have body, title-vs-title otherwise) ─────────────────
def test_pair_texts_uses_full_content_when_both_have_body():
    a = Topic(title="Hotel", description="manage hotel bookings and rooms")
    b = Topic(title="Resort", description="manage resort reservations")
    ta, tb = sc._pair_texts(a, b)
    assert "bookings" in ta and "reservations" in tb   # descriptions included


def test_pair_texts_falls_back_to_title_when_one_is_title_only():
    a = Topic(title="Hotel management", description="a long description of the hotel system")
    b = Topic(title="Pharmacy management")   # title only
    ta, tb = sc._pair_texts(a, b)
    assert ta == "Hotel management" and tb == "Pharmacy management"   # descriptions NOT used


# ── semester parsing + recent filter ────────────────────────────────────────────────
def test_semester_key_orders_seasons_and_years():
    assert sc._semester_key("Summer 2026") > sc._semester_key("Spring 2026")
    assert sc._semester_key("Spring 2026") > sc._semester_key("Fall 2025")
    assert sc._semester_key("garbage") is None


def test_recent_semesters_picks_two_most_recent():
    topics = [Topic(semester=s) for s in ("Fall 2025", "Spring 2026", "Summer 2026", "Fall 2025")]
    assert sc._recent_semesters(topics, 2) == {"Summer 2026", "Spring 2026"}


def test_recent_semesters_ignores_unparseable():
    topics = [Topic(semester="2024-1"), Topic(semester="Spring 2026")]
    assert sc._recent_semesters(topics, 2) == {"Spring 2026"}


# ── build_lexical_scorer fits on full content ───────────────────────────────────────
def test_lexical_uses_full_content():
    corpus = [
        Topic(title="Hotel app", description="booking rooms guests reservation front desk"),
        Topic(title="Hotel app", description="pharmacy inventory medicine prescription stock"),
    ]
    model = build_lexical_scorer(corpus)
    # Same title, very different descriptions → full-content lexical must be < 1.0 (title-only would be 1.0).
    a = sc._content_text(corpus[0])
    b = sc._content_text(corpus[1])
    assert model.similarity(a, b) < 0.95
    assert model.similarity(a, a) == 1.0
