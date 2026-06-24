from sqlalchemy.orm import Session

from app.utils.score_calculator import calculate_similarity_for_new


class SimilarityService:
    def __init__(self, db: Session):
        self.db = db

    def run_for_new(self, thesis_ids: list[int]) -> None:
        calculate_similarity_for_new(self.db, thesis_ids)
