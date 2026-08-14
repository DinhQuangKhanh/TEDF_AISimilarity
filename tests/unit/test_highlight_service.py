"""Unit tests for the field-aligned highlight service (P5.1).

Guards the two regressions the reviewer hit: (1) generic/stop words leaking into the painted
spans, and (2) the shape the frontend renders. The semantic angle needs SBERT; these tests only
assert structure + the distinctive-term filtering, so they run without asserting model output.
"""

from __future__ import annotations

from app.services.highlight_service import (
    _HIGHLIGHT_STOPWORDS,
    _distinctive_terms,
    compute_highlights,
)


class _Tag:
    def __init__(self, name: str):
        self.name = name


class _Topic:
    def __init__(self, **kw):
        self.title = kw.get("title")
        self.description = kw.get("description")
        self.objectives = kw.get("objectives")
        self.scope = kw.get("scope")
        self.expected_result = kw.get("expected_result")
        self.technologies = [_Tag(t) for t in kw.get("technologies", [])]


def test_compute_highlights_field_aligned_shape():
    a = _Topic(title="Hotel booking system", description="Manage hotel room bookings and payments.")
    b = _Topic(title="Restaurant booking system", description="Manage restaurant table bookings and payments.")
    out = compute_highlights(a, b, lexical_model=None)

    assert set(out.keys()) == {"fields"}
    assert isinstance(out["fields"], list)
    valid_fields = {"title", "description", "objectives", "scope", "technologies", "expectedResults"}
    valid_angles = {"semantic", "lexical", "structural", "domain"}
    for field in out["fields"]:
        assert set(field.keys()) == {"field", "angle", "score", "a", "b"}
        assert field["field"] in valid_fields
        assert field["angle"] in valid_angles
        for span in field["a"] + field["b"]:
            assert set(span.keys()) == {"text", "angle"}
            assert span["angle"] in valid_angles


def test_generic_and_stop_words_are_never_painted():
    # Both sides share many generic SE words; NONE may surface as a lexical span.
    a = _Topic(description="This system allows users to manage information and access services.")
    b = _Topic(description="This system provides users with data and services through a platform.")
    out = compute_highlights(a, b, lexical_model=None)

    painted = {
        span["text"].lower()
        for field in out["fields"]
        for span in field["a"] + field["b"]
        if span["angle"] == "lexical"
    }
    generic = {"is", "this", "user", "users", "service", "services", "system",
               "management", "manage", "information", "data", "access", "provide"}
    assert painted.isdisjoint(generic)


def test_distinctive_terms_drop_stopwords_and_short():
    a = "Hospital pharmacy inventory and prescription workflow"
    b = "Clinic pharmacy inventory and prescription workflow"
    terms = _distinctive_terms(a, b, lexical_model=None)

    assert "pharmacy" in terms                      # distinctive shared noun kept
    assert all(t not in _HIGHLIGHT_STOPWORDS for t in terms)
    assert all(len(t) >= 3 for t in terms)


def test_idf_floor_drops_corpus_wide_terms():
    class _FakeModel:
        def idf(self, term: str) -> float:
            return 3.0 if term == "pharmacy" else 0.5   # everything else below the 1.2 floor

    terms = _distinctive_terms(
        "Hospital pharmacy inventory prescription",
        "Clinic pharmacy inventory prescription",
        _FakeModel(),
    )
    assert "pharmacy" in terms
    assert "inventory" not in terms and "prescription" not in terms


def test_technologies_alignment_is_structural():
    a = _Topic(title="A", technologies=["ReactJS", "Node.js", "MongoDB"])
    b = _Topic(title="B", technologies=["ReactJS", "Node.js", "PostgreSQL"])
    out = compute_highlights(a, b, lexical_model=None)

    tech = next((f for f in out["fields"] if f["field"] == "technologies"), None)
    assert tech is not None
    assert tech["angle"] == "structural"
    assert tech["a"] and tech["b"]
    assert all(s["angle"] == "structural" for s in tech["a"] + tech["b"])
