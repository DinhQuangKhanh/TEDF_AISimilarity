"""Integration tests for the demo analyze endpoint and the demo page."""

from fastapi.testclient import TestClient

import app.models  # noqa: F401 - register models with Base.metadata
from app.database import Base, engine
from app.main import app

client = TestClient(app)


def setup_function():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)


def teardown_function():
    Base.metadata.drop_all(bind=engine)


def _seed(thesis_id, title, domains, tech):
    client.post("/api/v1/theses", json={
        "thesis_id": thesis_id, "title": title,
        "description": f"A system for {title}.", "scope": "web app", "objectives": "manage",
        "expected_result": "a deployed app", "semester": "2024-1", "program": "SE",
        "domains": domains, "technologies": tech,
    })


def test_analyze_returns_full_pipeline_and_matches():
    _seed("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa", "Pharmacy Management System",
          ["Pharmacy"], ["React", "Node.js"])

    resp = client.post("/api/v1/similarity/analyze", json={
        "title": "Hotel Management System using React and Node.js",
        "technologies": ["React", "Node.js"],
    })
    assert resp.status_code == 200
    data = resp.json()["data"]

    ids = [s["id"] for s in data["steps"]]
    assert ids == ["preprocess", "semantic", "corpus", "scoring", "mddm"]
    assert data["corpusSize"] == 1
    assert len(data["topMatches"]) == 1
    top = data["topMatches"][0]
    assert set(top["breakdown"]) == {"semantic", "lexical", "structure", "domain"}
    assert top["is_structural_duplication"] is True   # same stack, different domain
    assert top["revision_suggestion"]


def test_analyze_empty_title_is_error_step():
    resp = client.post("/api/v1/similarity/analyze", json={"title": "   "})
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["steps"][0]["id"] == "preprocess"
    assert data["steps"][0]["status"] == "error"


def test_analyze_empty_corpus_reports_error_step():
    resp = client.post("/api/v1/similarity/analyze", json={"title": "Hotel with React"})
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert any(s["id"] == "corpus" and s["status"] == "error" for s in data["steps"])


def test_demo_page_is_served():
    resp = client.get("/demo")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "DASSF" in resp.text
    assert "Kiểm tra trùng lặp" in resp.text
