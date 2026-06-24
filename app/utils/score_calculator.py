import json

from sqlalchemy.orm import Session

from app.models.similarity import Similarity
from app.models.thesis import Thesis
from app.utils.text_cleaner import tokenize

WEIGHTS = {
    "semantic": 0.40,
    "lexical": 0.25,
    "structure": 0.20,
    "domain": 0.15,
}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def clamp(score: float) -> float:
    return max(0.0, min(1.0, round(score, 4)))


def level_for(score: float) -> str:
    if score >= 0.85:
        return "Critical"
    if score >= 0.70:
        return "High"
    if score >= 0.50:
        return "Medium"
    if score >= 0.30:
        return "Low"
    return "Very Low"


def calculate_scores(a: Thesis, b: Thesis) -> dict:
    semantic_a = tokenize(" ".join(filter(None, [a.title_en, a.title_vn, a.description])))
    semantic_b = tokenize(" ".join(filter(None, [b.title_en, b.title_vn, b.description])))
    lexical_a = {t.name.lower() for t in a.lexical_tags} | tokenize(a.title_en) | tokenize(a.title_vn)
    lexical_b = {t.name.lower() for t in b.lexical_tags} | tokenize(b.title_en) | tokenize(b.title_vn)
    structure_a = {s.name.lower() for s in a.structures} | tokenize(a.scope)
    structure_b = {s.name.lower() for s in b.structures} | tokenize(b.scope)
    domain_a = {d.name.lower() for d in a.domains} | {t.name.lower() for t in a.technologies}
    domain_b = {d.name.lower() for d in b.domains} | {t.name.lower() for t in b.technologies}

    semantic = clamp(jaccard(semantic_a, semantic_b))
    lexical = clamp(jaccard(lexical_a, lexical_b))
    structure = clamp(jaccard(structure_a, structure_b))
    domain = clamp(jaccard(domain_a, domain_b))
    overall = clamp(
        semantic * WEIGHTS["semantic"]
        + lexical * WEIGHTS["lexical"]
        + structure * WEIGHTS["structure"]
        + domain * WEIGHTS["domain"]
    )
    reasons = []
    if domain > 0:
        reasons.append("same business domain or shared technologies")
    if structure > 0:
        reasons.append("similar architecture or scope")
    if lexical > 0:
        reasons.append("shared lexical tags or title terms")
    if semantic > 0:
        reasons.append("similar semantic content")
    return {
        "semantic_score": semantic,
        "lexical_score": lexical,
        "structure_score": structure,
        "domain_score": domain,
        "overall_score": overall,
        "level": level_for(overall),
        "reason": reasons,
    }


def calculate_similarity_for_new(db: Session, new_ids: list[int]) -> None:
    all_theses = db.query(Thesis).filter(Thesis.is_deleted.is_(False)).all()
    thesis_map = {thesis.thesis_id: thesis for thesis in all_theses}
    new_theses = [thesis_map[thesis_id] for thesis_id in new_ids if thesis_id in thesis_map]

    for new_thesis in new_theses:
        for other in all_theses:
            if new_thesis.thesis_id == other.thesis_id:
                continue
            thesis_a_id, thesis_b_id = sorted([new_thesis.thesis_id, other.thesis_id])
            exists = (
                db.query(Similarity)
                .filter(Similarity.thesis_a_id == thesis_a_id, Similarity.thesis_b_id == thesis_b_id)
                .first()
            )
            if exists:
                continue
            scores = calculate_scores(new_thesis, other)
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
