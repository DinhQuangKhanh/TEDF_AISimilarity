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


_STEP_IDS = ["input", "preprocess", "sedo", "semantic", "corpus", "scoring", "mddm", "decision"]


# ── error / early-exit branches (no SBERT needed) ───────────────────────────────────
def test_missing_title_is_an_error_step():
    result = analyze_topic(Topic(""), [Topic("Something")], None)
    assert _ids(result["steps"]) == ["input"]
    assert result["steps"][0]["status"] == "error"
    assert result["topMatches"] == []


def test_empty_corpus_stops_at_corpus_step():
    result = analyze_topic(Topic("Hotel Management with React"), [], None)
    ids = _ids(result["steps"])
    assert "corpus" in ids
    assert _by_id(result["steps"], "corpus")["status"] == "error"
    assert result["topMatches"] == []
    assert result["corpusSize"] == 0


def test_preprocess_emits_normalized_tokens():
    result = analyze_topic(Topic("Hotel Management Systems using ReactJS"), [], None)
    pre = _by_id(result["steps"], "preprocess")
    assert pre["status"] == "success"
    tokens = pre["output"]["tokens"]
    assert "react" in tokens          # reactjs folded onto the canonical concept
    assert "system" not in tokens     # generic stop-word dropped
    assert pre["output"]["tokenCount"] == len(tokens)


def test_sedo_step_recognizes_concepts():
    result = analyze_topic(Topic("Hotel Management System using React"), [], None)
    sedo = _by_id(result["steps"], "sedo")
    assert sedo["status"] == "success"
    assert "React" in sedo["output"]["tech"]
    assert "Hotel" in sedo["output"]["domains"]


def test_sedo_step_warns_when_no_concept():
    result = analyze_topic(Topic("Zzz Qqq Wwww"), [], None)
    assert _by_id(result["steps"], "sedo")["status"] == "warning"


def test_title_only_topic_is_flagged_in_the_input_step():
    full = analyze_topic(Topic("Hotel app", description="Manage bookings"), [], None)
    assert _by_id(full["steps"], "input")["output"]["matchMode"] == "full-content"
    bare = analyze_topic(Topic("Hotel app"), [], None)
    bare_step = _by_id(bare["steps"], "input")
    assert bare_step["output"]["matchMode"] == "title-only"
    assert bare_step["status"] == "warning"


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

    assert _ids(result["steps"]) == _STEP_IDS
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


def test_full_pipeline_flags_structural_duplicate(monkeypatch):
    # Exercises the SEDO structural-dup path (structure high + domain low). Pin capability OFF so the
    # test is deterministic regardless of .env: the DEFAULT running config uses the function-centric
    # capability dimension, under which two SHORT same-tech/different-domain titles do NOT share core
    # functions → correctly NOT flagged (see the campus-event demo, ICTA §H).
    monkeypatch.setattr("app.core.config.CAPABILITY_STRUCTURAL", False)
    corpus = [Topic("Pharmacy Management System using React and Node.js", semester="Spring 2026")]
    result = analyze_topic(Topic("Hotel Management System using React and Node.js"), corpus, None)
    top = result["topMatches"][0]
    assert top["is_structural_duplication"] is True
    assert "React" in top["shared_concepts"]["tech"]
    # the trace must agree with the verdict it explains
    assert _by_id(result["steps"], "decision")["output"]["structuralDuplication"] is True


def test_mddm_step_shows_the_arithmetic_that_produced_the_score():
    """The fusion step is the one readers are asked to trust — its printed terms must be the
    actual α·S_sem … δ·S_dom that sum to the reported overall score."""
    corpus = [Topic("Pharmacy Management System using React and Node.js", semester="Spring 2026")]
    result = analyze_topic(Topic("Hotel Management System using React and Node.js"), corpus, None)
    out = _by_id(result["steps"], "mddm")["output"]
    top = result["topMatches"][0]

    assert [t["dim"] for t in out["terms"]] == ["semantic", "lexical", "structure", "domain"]
    for term in out["terms"]:
        assert term["score"] == top["breakdown"][term["dim"]]
        assert abs(term["weight"] * term["score"] - term["product"]) < 1e-4
    assert abs(sum(t["product"] for t in out["terms"]) - out["overall"]) < 1e-3
    assert out["overall"] == top["overall_score"]
    assert abs(sum(t["weight"] for t in out["terms"]) - 1.0) < 1e-6
