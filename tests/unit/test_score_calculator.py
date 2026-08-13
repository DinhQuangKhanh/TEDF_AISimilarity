from app.models.classification import Domain, LexicalTag, SemanticCategory, StructureType, Tech
from app.models.thesis import Thesis
from app.utils.score_calculator import (
    WEIGHTS,
    action_for,
    calculate_scores,
    is_structural_duplication,
    level_for,
    weighted_jaccard,
)

LEVELS = {"Critical", "High", "Moderate", "Low"}


def build_thesis(title: str):
    # Uses terms the SEDO ontology recognizes (React, Hotel) so the structural
    # and domain dimensions are exercised.
    thesis = Thesis(
        title=title,
        description="a hotel booking management system",
        scope="React frontend with PostgreSQL database and REST API",
        objectives="digitize hotel booking and manage rooms",
        expected_result="a deployed web application",
    )
    thesis.domains = [Domain(name="Hotel")]
    thesis.semantics = [SemanticCategory(name="Management System")]
    thesis.structures = [StructureType(name="Web Application")]
    thesis.lexical_tags = [LexicalTag(name="order")]
    thesis.technologies = [Tech(name="React")]
    return thesis


def test_calculate_scores_range():
    left = build_thesis("Project Alpha")
    right = build_thesis("Project Alpha")
    scores = calculate_scores(left, right)
    assert 0.0 <= scores["semantic_score"] <= 1.0
    assert 0.0 <= scores["overall_score"] <= 1.0
    assert scores["level"] in LEVELS


def test_identical_topics_are_critical():
    scores = calculate_scores(build_thesis("Project Alpha"), build_thesis("Project Alpha"))
    assert scores["overall_score"] == 1.0
    assert scores["level"] == "Critical"
    assert scores["action"] == "Reject"


def test_weights_match_paper():
    # DASSF paper Eq. 1: alpha=0.30, beta=0.20, gamma=0.30, delta=0.20
    assert WEIGHTS == {"semantic": 0.30, "lexical": 0.20, "structure": 0.30, "domain": 0.20}
    assert round(sum(WEIGHTS.values()), 6) == 1.0


def test_level_for_matches_table_3():
    assert level_for(0.90) == "Critical"
    assert level_for(0.85) == "Critical"
    assert level_for(0.70) == "High"
    assert level_for(0.65) == "High"
    assert level_for(0.50) == "Moderate"
    assert level_for(0.40) == "Moderate"
    assert level_for(0.10) == "Low"


def test_action_for_matches_table_3():
    assert action_for("Low") == "Accept"
    assert action_for("Moderate") == "Warn; committee reviews"
    assert action_for("High") == "Require substantial revision"
    assert action_for("Critical") == "Reject"


def test_is_structural_duplication():
    # Same tech stack (high structural), different business domain (low domain).
    assert is_structural_duplication(0.95, 0.20)
    # Same domain -> not the structural class.
    assert not is_structural_duplication(0.95, 0.80)
    # Different stack -> not the structural class.
    assert not is_structural_duplication(0.30, 0.20)


def test_reasons_are_not_contradictory():
    # A structural duplication (different domain) must never also claim "same business domain".
    left = build_thesis("Hotel Management System")
    right = build_thesis("Pharmacy Management System")
    right.description = "a pharmacy inventory management system"
    right.objectives = "digitize pharmacy sales and manage medicine stock"
    right.domains = [Domain(name="Pharmacy")]
    scores = calculate_scores(left, right)
    assert scores["structural_duplication"] is True
    assert "same business domain" not in scores["reason"]
    assert "same tech stack with a different business domain" in scores["reason"]


def test_weighted_jaccard_without_idf_is_plain_jaccard():
    assert weighted_jaccard({"x", "y"}, {"y", "z"}, None) == 1 / 3
    assert weighted_jaccard(set(), set(), None) == 0.0


def test_weighted_jaccard_uses_idf():
    idf = {"common": 1.0, "rare": 5.0}
    # intersection weight = 1.0 ("common"); union weight = 1.0 + 5.0
    assert weighted_jaccard({"common", "rare"}, {"common"}, idf) == 1.0 / 6.0
