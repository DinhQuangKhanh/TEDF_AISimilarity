"""Unit tests for the per-field 'explain duplication' service (P5).

Covers the two guarantees: (1) explanations are grounded in the highlight evidence (template mentions
the exact spans/terms/concepts), and (2) the feature works with OR without an LLM (LLM JSON is used
when valid, otherwise a deterministic template). The LLM + highlight computation are patched so these
run fast and offline.
"""

from __future__ import annotations

from app.services import explain_service


# ── template (grounded, pure) ─────────────────────────────────────────────────────────
def test_template_semantic_is_grounded_in_the_passage():
    ev = {"field": "description", "label": "Mô tả", "angle": "semantic", "score": 0.82,
          "a": "manage boat docks and tours", "b": "manage users and services", "terms": [], "concepts": []}
    s = explain_service._template_explanation(ev)
    assert "Mô tả" in s and "82%" in s and "manage boat docks and tours" in s


def test_template_structural_lists_concepts():
    ev = {"field": "technologies", "label": "Công nghệ", "angle": "structural", "score": None,
          "a": None, "b": None, "terms": [], "concepts": ["React", "SQL Server"]}
    s = explain_service._template_explanation(ev)
    assert "React" in s and "SQL Server" in s


def test_template_lexical_lists_terms():
    ev = {"field": "scope", "label": "Phạm vi", "angle": "lexical", "score": None,
          "a": None, "b": None, "terms": ["booking", "dispatcher"], "concepts": []}
    s = explain_service._template_explanation(ev)
    assert "booking" in s and "dispatcher" in s


# ── JSON extraction ───────────────────────────────────────────────────────────────────
def test_parse_json_object_extracts_and_rejects():
    assert explain_service._parse_json_object('rác {"a": "b"} rác') == {"a": "b"}
    assert explain_service._parse_json_object("không có json") is None
    assert explain_service._parse_json_object("{hỏng}") is None


# ── explain(): LLM used when valid, template otherwise ────────────────────────────────
_EV = [{"field": "description", "label": "Mô tả", "angle": "semantic", "score": 0.8,
        "a": "manage pet care", "b": "manage pet care and appointments", "terms": ["pet"], "concepts": []}]


def test_explain_uses_llm_json_when_valid(monkeypatch):
    monkeypatch.setattr(explain_service, "build_field_evidence", lambda q, m: _EV)
    monkeypatch.setattr(explain_service, "call_llm", lambda prompt: '{"description": "Giải thích do LLM viết."}')
    out = explain_service.explain(None, None)
    assert out == [{"field": "description", "angle": "semantic", "score": 0.8, "explanation": "Giải thích do LLM viết."}]


def test_explain_falls_back_to_template_when_llm_none(monkeypatch):
    monkeypatch.setattr(explain_service, "build_field_evidence", lambda q, m: _EV)
    monkeypatch.setattr(explain_service, "call_llm", lambda prompt: None)  # LLM unavailable
    out = explain_service.explain(None, None)
    assert out[0]["field"] == "description"
    assert "manage pet care" in out[0]["explanation"]  # grounded template, not empty


def test_explain_falls_back_when_llm_returns_garbage(monkeypatch):
    monkeypatch.setattr(explain_service, "build_field_evidence", lambda q, m: _EV)
    monkeypatch.setattr(explain_service, "call_llm", lambda prompt: "xin chào, không phải json")
    out = explain_service.explain(None, None)
    assert "manage pet care" in out[0]["explanation"]


def test_explain_skips_llm_when_disabled(monkeypatch):
    called = {"n": 0}

    def _fake_llm(_):
        called["n"] += 1
        return '{"description": "x"}'

    monkeypatch.setattr(explain_service, "build_field_evidence", lambda q, m: _EV)
    monkeypatch.setattr(explain_service, "call_llm", _fake_llm)
    out = explain_service.explain(None, None, use_llm=False)
    assert called["n"] == 0                       # never calls the LLM
    assert "manage pet care" in out[0]["explanation"]


def test_explain_empty_evidence_returns_empty(monkeypatch):
    monkeypatch.setattr(explain_service, "build_field_evidence", lambda q, m: [])
    assert explain_service.explain(None, None) == []
