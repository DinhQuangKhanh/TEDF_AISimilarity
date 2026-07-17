import json
import math

from sqlalchemy.orm import Session

from app.models.similarity import Similarity
from app.models.thesis import Thesis
from app.utils.text_cleaner import tokenize

# MDDM fusion weights (DASSF paper, Eq. 1): alpha=0.30, beta=0.20, gamma=0.30, delta=0.20.
WEIGHTS = {
    "semantic": 0.30,
    "lexical": 0.20,
    "structure": 0.30,
    "domain": 0.20,
}

# Structural-duplication rule (paper Sect. 3.5): S_str >= TAU_STR and S_dom < TAU_DOM.
# The paper does not fix these numerically; tune them on a labeled set.
TAU_STR = 0.65
TAU_DOM = 0.40

# Four-level decision scale (paper Table 3).
_ACTIONS = {
    "Low": "Accept",
    "Moderate": "Warn; committee reviews",
    "High": "Require substantial revision",
    "Critical": "Reject",
}


def _content_text(thesis: Thesis) -> str:
    """Concatenate all five topic fields (paper Sect. 3.4)."""
    return " ".join(
        filter(
            None,
            [thesis.title, thesis.description, thesis.scope, thesis.objectives, thesis.expected_result],
        )
    )


def jaccard(a: set[str], b: set[str]) -> float:
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def weighted_jaccard(a: set[str], b: set[str], idf: dict[str, float] | None = None) -> float:
    """TF-IDF weighted Jaccard; idf=None falls back to plain Jaccard."""
    union = a | b
    if not union:
        return 0.0
    if idf is None:
        return len(a & b) / len(union)
    numerator = sum(idf.get(token, 1.0) for token in a & b)
    denominator = sum(idf.get(token, 1.0) for token in union)
    return numerator / denominator if denominator else 0.0


def build_idf(theses: list[Thesis]) -> dict[str, float]:
    """Smoothed IDF over the five-field text of the whole corpus."""
    total = len(theses)
    document_frequency: dict[str, int] = {}
    for thesis in theses:
        for token in tokenize(_content_text(thesis)):
            document_frequency[token] = document_frequency.get(token, 0) + 1
    return {
        token: math.log((total + 1) / (count + 1)) + 1.0
        for token, count in document_frequency.items()
    }


def clamp(score: float) -> float:
    return max(0.0, min(1.0, round(score, 4)))


def level_for(score: float) -> str:
    if score >= 0.85:
        return "Critical"
    if score >= 0.65:
        return "High"
    if score >= 0.40:
        return "Moderate"
    return "Low"


def action_for(level: str) -> str:
    return _ACTIONS.get(level, "")


def is_structural_duplication(structure_score: float, domain_score: float) -> bool:
    """Same tech stack, different business domain (paper Sect. 3.5)."""
    return structure_score >= TAU_STR and domain_score < TAU_DOM


def calculate_scores(a: Thesis, b: Thesis, idf: dict[str, float] | None = None) -> dict:
    # Semantic and lexical both read all five fields (paper Table 2); they differ in weighting.
    content_a = tokenize(_content_text(a))
    content_b = tokenize(_content_text(b))

    # Structural = tech stack + methodology, read from scope + description.
    structure_a = (
        {item.name.lower() for item in a.structures}
        | {item.name.lower() for item in a.technologies}
        | tokenize(a.scope)
        | tokenize(a.description)
    )
    structure_b = (
        {item.name.lower() for item in b.structures}
        | {item.name.lower() for item in b.technologies}
        | tokenize(b.scope)
        | tokenize(b.description)
    )

    # Domain = business domain, read from description + objectives.
    domain_a = {item.name.lower() for item in a.domains} | tokenize(a.description) | tokenize(a.objectives)
    domain_b = {item.name.lower() for item in b.domains} | tokenize(b.description) | tokenize(b.objectives)

    semantic = clamp(jaccard(content_a, content_b))
    lexical = clamp(weighted_jaccard(content_a, content_b, idf))
    structure = clamp(jaccard(structure_a, structure_b))
    domain = clamp(jaccard(domain_a, domain_b))
    overall = clamp(
        semantic * WEIGHTS["semantic"]
        + lexical * WEIGHTS["lexical"]
        + structure * WEIGHTS["structure"]
        + domain * WEIGHTS["domain"]
    )
    level = level_for(overall)
    structural_dup = is_structural_duplication(structure, domain)

    reasons = []
    if structural_dup:
        reasons.append("same tech stack with a different business domain")
    if domain > 0:
        reasons.append("same business domain")
    if structure > 0:
        reasons.append("similar architecture or scope")
    if lexical > 0:
        reasons.append("shared weighted terms across fields")
    if semantic > 0:
        reasons.append("similar semantic content")
    return {
        "semantic_score": semantic,
        "lexical_score": lexical,
        "structure_score": structure,
        "domain_score": domain,
        "overall_score": overall,
        "level": level,
        "action": action_for(level),
        "structural_duplication": structural_dup,
        "reason": reasons,
    }


def calculate_similarity_for_new(db: Session, new_ids: list[int]) -> None:
    all_theses = db.query(Thesis).filter(Thesis.is_deleted.is_(False)).all()
    thesis_map = {thesis.thesis_id: thesis for thesis in all_theses}
    new_theses = [thesis_map[thesis_id] for thesis_id in new_ids if thesis_id in thesis_map]

    # IDF is corpus-level, so build it once per run.
    idf = build_idf(all_theses)

    seen: set[tuple[int, int]] = set()
    for new_thesis in new_theses:
        for other in all_theses:
            if new_thesis.thesis_id == other.thesis_id:
                continue
            thesis_a_id, thesis_b_id = sorted([new_thesis.thesis_id, other.thesis_id])
            pair = (thesis_a_id, thesis_b_id)
            if pair in seen:
                continue
            seen.add(pair)
            exists = (
                db.query(Similarity)
                .filter(Similarity.thesis_a_id == thesis_a_id, Similarity.thesis_b_id == thesis_b_id)
                .first()
            )
            if exists:
                continue
            scores = calculate_scores(new_thesis, other, idf)
            db.add(
                Similarity(
                    thesis_a_id=thesis_a_id,
                    thesis_b_id=thesis_b_id,
                    semantic_score=scores["semantic_score"],
                    lexical_score=scores["lexical_score"],
                    structure_score=scores["structure_score"],
                    domain_score=scores["domain_score"],
                    overall_score=scores["overall_score"],
                    level=scores["level"],
                    reason=json.dumps(scores["reason"]),
                )
            )
    db.flush()
