import json
import uuid

from sqlalchemy.orm import Session

from app.repositories.similarity_repository import SimilarityRepository
from app.utils.score_calculator import action_for, calculate_similarity_for_new, is_structural_duplication


def serialize_similarity(row) -> dict:
    """One similarity row plus the decision fields derived from it (paper Table 3 / Sect. 3.5)."""
    return {
        "sim_id": row.sim_id,
        "thesis_a_id": str(row.thesis_a_id),
        "thesis_b_id": str(row.thesis_b_id),
        "semantic_score": row.semantic_score,
        "lexical_score": row.lexical_score,
        "structure_score": row.structure_score,
        "domain_score": row.domain_score,
        "overall_score": row.overall_score,
        "level": row.level,
        "action": action_for(row.level),
        "is_structural_duplication": is_structural_duplication(row.structure_score, row.domain_score, row.overall_score),
        "reason": json.loads(row.reason) if row.reason else [],
    }


class SimilarityService:
    def __init__(self, db: Session):
        self.db = db
        self.repo = SimilarityRepository(db)

    def run_for_new(self, thesis_ids: list[uuid.UUID]) -> None:
        calculate_similarity_for_new(self.db, thesis_ids)

    def results_for(self, thesis_ids: list[uuid.UUID]) -> list[dict]:
        """Every stored pair touching any of these theses, highest score first."""
        seen: set[int] = set()
        items: list[dict] = []
        for thesis_id in thesis_ids:
            for row in self.repo.list_for_thesis(thesis_id):
                if row.sim_id in seen:
                    continue
                seen.add(row.sim_id)
                items.append(serialize_similarity(row))
        items.sort(key=lambda item: item["overall_score"], reverse=True)
        return items
