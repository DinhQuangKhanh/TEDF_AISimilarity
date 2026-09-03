"""Integration tests for the demo analyze endpoint and the demo page."""

from fastapi.testclient import TestClient

import app.models  # noqa: F401 - register models with Base.metadata
from app.database import Base, engine
from app.main import app

client = TestClient(app)

# The demo trace is the feature under test here: eight named steps, in this order.
_STEP_IDS = ["input", "preprocess", "sedo", "semantic", "corpus", "scoring", "mddm", "decision"]


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

    assert [s["id"] for s in data["steps"]] == _STEP_IDS
    assert data["corpusSize"] == 53                         # SP26 (40) + SU26 (13)
    assert 1 <= len(data["topMatches"]) <= 5
    top = data["topMatches"][0]
    assert set(top["breakdown"]) == {"semantic", "lexical", "structure", "domain"}
    assert "revision_suggestion" not in top          # removed: editing is the proposer's job, not the evaluator's
    assert top["otherSemester"] in {"Spring 2026", "Summer 2026"}
    # matched topic content (side-by-side) + per-dimension highlight spans
    assert top["other"]["title"] and "title" in top["other"]
    # field-aligned highlights: only fields with a real overlap, each carrying angle + typed spans
    assert set(top["highlights"]) == {"fields"}
    fields = top["highlights"]["fields"]
    assert isinstance(fields, list)
    valid_fields = {"title", "description", "objectives", "scope", "technologies", "expectedResults"}
    valid_angles = {"semantic", "lexical", "structural", "domain"}
    for fh in fields:
        assert set(fh.keys()) == {"field", "angle", "score", "a", "b"}
        assert fh["field"] in valid_fields
        assert fh["angle"] in valid_angles
        assert all(set(sp.keys()) == {"text", "angle"} for sp in fh["a"] + fh["b"])
        assert all(sp["angle"] in valid_angles for sp in fh["a"] + fh["b"])
    assert data["query"]["title"].startswith("EcoTrack")


def test_analyze_title_only_topic_still_works():
    # A topic with only a title falls back to title-based comparison (paper mode).
    resp = client.post("/api/v1/similarity/analyze", json={"title": "Hotel Management System using React"})
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["corpusSize"] == 53
    assert [s["id"] for s in data["steps"]] == _STEP_IDS


def test_analyze_empty_title_is_error_step():
    resp = client.post("/api/v1/similarity/analyze", json={"title": "   "})
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["steps"][0]["id"] == "input"
    assert data["steps"][0]["status"] == "error"


def test_every_step_carries_an_input_output_pair():
    """The demo's whole point is that each step is inspectable — a step with no output would
    render as an empty card."""
    resp = client.post("/api/v1/similarity/analyze",
                       json={"title": "Hotel Management System using React and Node.js"})
    steps = resp.json()["data"]["steps"]
    assert len(steps) == len(_STEP_IDS)
    for step in steps:
        assert step["output"], f"step {step['id']} has no output"
        assert step["detail"], f"step {step['id']} has no explanation"
    # the fusion step must show the arithmetic, not just the total
    mddm = next(s for s in steps if s["id"] == "mddm")
    terms = mddm["output"]["terms"]
    assert [t["dim"] for t in terms] == ["semantic", "lexical", "structure", "domain"]
    assert abs(sum(t["product"] for t in terms) - mddm["output"]["overall"]) < 1e-3


def test_model_info_reports_weights_and_their_source():
    data = client.get("/api/v1/similarity/model-info").json()["data"]
    assert set(data["weights"]) == {"semantic", "lexical", "structure", "domain"}
    assert abs(sum(data["weights"].values()) - 1.0) < 1e-6
    assert data["weightsSource"]
    assert [t["level"] for t in data["levelThresholds"]] == ["Critical", "High", "Moderate", "Low"]
    assert set(data["backends"]) == {"semantic", "lexical"}
    # the grid-search log is optional (only present after tools/trace_tuning.py has run)
    if data["tuning"] is not None:
        assert data["tuning"]["gridPoints"] > 0
        assert data["tuning"]["leaderboard"][0]["rank"] == 1


def test_demo_page_is_served():
    resp = client.get("/demo")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "DASSF" in resp.text
    assert "Kiểm tra trùng lặp" in resp.text
