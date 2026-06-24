from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.models.similarity import Similarity


class SimilarityRepository:
    def __init__(self, db: Session):
        self.db = db

    def list_for_thesis(self, thesis_id: int):
        return (
            self.db.query(Similarity)
            .filter(or_(Similarity.thesis_a_id == thesis_id, Similarity.thesis_b_id == thesis_id))
            .order_by(Similarity.overall_score.desc())
            .all()
        )
