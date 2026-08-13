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


def test_analyze_uses_recent_capstone_corpus_dbfree():
    # The demo now compares against Spring 2026 + Summer 2026 (loaded from JSON), no DB seeding.
    resp = client.post("/api/v1/similarity/analyze", json={
        "title": "EcoTrack - Smart Household Waste Management System using React and ASP.NET Core",
        "description": "A platform to track, classify and manage household waste with recycling stats.",
        "scope": "User management, waste classification, collection scheduling, reporting.",
        "objectives": "Centralize household waste activities; encourage recycling.",
        "expected_result": "A web app with user and admin interfaces.",
        "technologies": ["React", "TypeScript", "ASP.NET Core", "PostgreSQL", "Redis", "Docker"],
    })
    assert resp.status_code == 200
    data = resp.json()["data"]

    assert [s["id"] for s in data["steps"]] == ["preprocess", "semantic", "corpus", "scoring", "mddm"]
    assert data["corpusSize"] == 53                         # SP26 (40) + SU26 (13)
    assert 1 <= len(data["topMatches"]) <= 5
    top = data["topMatches"][0]
    assert set(top["breakdown"]) == {"semantic", "lexical", "structure", "domain"}
    assert top["revision_suggestion"]
    assert top["otherSemester"] in {"Spring 2026", "Summer 2026"}


def test_analyze_title_only_topic_still_works():
    # A topic with only a title falls back to title-based comparison (paper mode).
    resp = client.post("/api/v1/similarity/analyze", json={"title": "Hotel Management System using React"})
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["corpusSize"] == 53
    assert [s["id"] for s in data["steps"]] == ["preprocess", "semantic", "corpus", "scoring", "mddm"]


def test_analyze_empty_title_is_error_step():
    resp = client.post("/api/v1/similarity/analyze", json={"title": "   "})
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["steps"][0]["id"] == "preprocess"
    assert data["steps"][0]["status"] == "error"


def test_demo_page_is_served():
    resp = client.get("/demo")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "DASSF" in resp.text
    assert "Kiểm tra trùng lặp" in resp.text
