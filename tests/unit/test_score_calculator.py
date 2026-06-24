from app.models.classification import Domain, LexicalTag, SemanticCategory, StructureType, Tech
from app.models.thesis import Thesis
from app.utils.score_calculator import calculate_scores, level_for


def build_thesis(title: str):
    thesis = Thesis(title_en=title, description="demo", scope="web platform")
    thesis.domains = [Domain(name="E-commerce")]
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
    assert scores["level"] in {"Critical", "High", "Medium", "Low", "Very Low"}


def test_level_for():
    assert level_for(0.9) == "Critical"
    assert level_for(0.1) == "Very Low"
