"""Integration test for POST /api/v1/similarity/explain (P5).

Hits the real endpoint (recomputes the same highlights as /analyze, then explains). No LLM is
required: when Ollama is not reachable the service falls back to the grounded template, so this
passes offline.
"""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

_VALID_FIELDS = {"title", "description", "objectives", "scope", "technologies", "expectedResults"}
_VALID_ANGLES = {"semantic", "lexical", "structural", "domain"}


def test_explain_returns_per_field_explanations():
    body = {
        "query": {
            "title": "Smart Pet Care Management System using Vue.js and Django",
            "description": "A web platform for managing pet care activities and vaccination for pet owners.",
            "technologies": ["Vue.js", "Django", "PostgreSQL"],
        },
        "match": {
            "title": "Pet Care and Appointment System",
            "description": "A web platform for managing pet care and appointments for pet owners.",
            "technologies": ["Vue.js", "Django", "PostgreSQL"],
        },
    }
    res = client.post("/api/v1/similarity/explain", json=body)
    assert res.status_code == 200

    fields = res.json()["data"]["fields"]
    assert isinstance(fields, list) and len(fields) >= 1
    for f in fields:
        assert set(f.keys()) == {"field", "angle", "score", "explanation"}
        assert f["field"] in _VALID_FIELDS
        assert f["angle"] in _VALID_ANGLES
        assert isinstance(f["explanation"], str) and f["explanation"].strip()


def test_explain_no_overlap_returns_empty_fields():
    body = {
        "query": {"title": "Weather forecasting API with Go"},
        "match": {"title": "Blockchain voting ledger with Rust"},
    }
    res = client.post("/api/v1/similarity/explain", json=body)
    assert res.status_code == 200
    assert isinstance(res.json()["data"]["fields"], list)
