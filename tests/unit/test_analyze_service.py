"""Tests for the dry-run pipeline analyzer (app/services/analyze_service.py)."""

from app.services.analyze_service import analyze_topic


class Tag:
    def __init__(self, name):
        self.name = name


class Topic:
    def __init__(self, title, description="", scope="", objectives="", expected_result="",
                 technologies=(), semester=None):
        self.title = title
        self.description = description
        self.scope = scope
        self.objectives = objectives
        self.expected_result = expected_result
        self.technologies = [Tag(t) for t in technologies]
        self.structures = []
        self.domains = []
        self.semester = semester


def _ids(steps):
    return [s["id"] for s in steps]


def _by_id(steps, sid):
    return next(s for s in steps if s["id"] == sid)


# ── error / early-exit branches (no SBERT needed) ───────────────────────────────────
def test_missing_title_is_an_error_step():
    result = analyze_topic(Topic(""), [Topic("Something")], None)
    assert _ids(result["steps"]) == ["preprocess"]
    assert result["steps"][0]["status"] == "error"
    assert result["topMatches"] == []


def test_empty_corpus_stops_at_corpus_step():
    result = analyze_topic(Topic("Hotel Management with React"), [], None)
    ids = _ids(result["steps"])
    assert "corpus" in ids
    assert _by_id(result["steps"], "corpus")["status"] == "error"
    assert result["topMatches"] == []
    assert result["corpusSize"] == 0


def test_preprocess_recognizes_concepts():
    result = analyze_topic(Topic("Hotel Management System using React"), [], None)
    pre = _by_id(result["steps"], "preprocess")
    assert pre["status"] == "success"
    assert "React" in pre["output"]["tech"]
    assert "Hotel" in pre["output"]["domains"]


def test_preprocess_warns_when_no_concept():
    result = analyze_topic(Topic("Zzz Qqq Wwww"), [], None)
    pre = _by_id(result["steps"], "preprocess")
    assert pre["status"] == "warning"


def test_semantic_step_reports_backend():
    result = analyze_topic(Topic("Hotel with React"), [], None, semantic_backend="jaccard-fallback")
    sem = _by_id(result["steps"], "semantic")
    assert sem["status"] == "warning"
    assert sem["output"]["backend"] == "jaccard-fallback"


# ── full pipeline (loads SBERT once) ────────────────────────────────────────────────
def test_full_pipeline_returns_ranked_matches():
    corpus = [
        Topic("Pharmacy Management System using React and Node.js", semester="Spring 2026"),
        Topic("Weather forecasting with Python", semester="Fall 2025"),
        Topic("Hotel booking platform using React and Node.js", semester="Summer 2026"),
    ]
    result = analyze_topic(Topic("Hotel Management System using React and Node.js"), corpus, None, top_k=3)

    assert _ids(result["steps"]) == ["preprocess", "semantic", "corpus", "scoring", "mddm"]
    assert all(s["status"] in ("success", "warning") for s in result["steps"])
    assert result["corpusSize"] == 3

    matches = result["topMatches"]
    assert len(matches) == 3
    # ranked by overall score, descending
    overalls = [m["overall_score"] for m in matches]
    assert overalls == sorted(overalls, reverse=True)
    # every match carries a full explanation
    for m in matches:
        assert set(m["breakdown"]) == {"semantic", "lexical", "structure", "domain"}
        assert m["level"] in {"Low", "Moderate", "High", "Critical"}
        assert "revision_suggestion" not in m
        assert "otherTitle" in m


def test_full_pipeline_flags_structural_duplicate():
    # same stack (React/Node), different domain → the paper's structural-duplication case
    corpus = [Topic("Pharmacy Management System using React and Node.js", semester="Spring 2026")]
    result = analyze_topic(Topic("Hotel Management System using React and Node.js"), corpus, None)
    top = result["topMatches"][0]
    assert top["is_structural_duplication"] is True
    assert "React" in top["shared_concepts"]["tech"]
