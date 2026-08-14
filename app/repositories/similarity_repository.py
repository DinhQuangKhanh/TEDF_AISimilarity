import uuid

from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.models.similarity import Similarity


class SimilarityRepository:
    def __init__(self, db: Session):
        self.db = db

    def list_for_thesis(self, thesis_id: uuid.UUID):
        return (
            self.db.query(Similarity)
            .filter(or_(Similarity.thesis_a_id == thesis_id, Similarity.thesis_b_id == thesis_id))
            .order_by(Similarity.overall_score.desc())
            .all()
        )

    def get_pair(self, a_id: uuid.UUID, b_id: uuid.UUID):
        """The stored similarity row for an unordered pair, or None. Pairs are stored with the
        two ids sorted (see calculate_similarity_for_new), so sort the lookup the same way."""
        low, high = sorted([a_id, b_id])
        return (
            self.db.query(Similarity)
            .filter(Similarity.thesis_a_id == low, Similarity.thesis_b_id == high)
            .first()
        )
