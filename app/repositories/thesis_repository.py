from typing import Optional

from sqlalchemy import or_
from sqlalchemy.orm import Session, joinedload

from app.models.thesis import Thesis
from app.utils.duplicate_checker import combined_content, content_similarity


class ThesisRepository:
    def __init__(self, db: Session):
        self.db = db

    def find_near_duplicate_candidates(
        self,
        title: str | None,
        description: str | None,
        scope: str | None,
        objectives: str | None,
        expected_result: str | None,
        semester: str | None,
        program: str | None,
    ):
        query = self.db.query(Thesis).filter(Thesis.is_deleted.is_(False))
        if semester:
            query = query.filter(Thesis.semester == semester)
        if program:
            query = query.filter(Thesis.program == program)
        incoming = combined_content(title, description, scope, objectives, expected_result)
        candidates = []
        for thesis in query.all():
            existing = combined_content(
                thesis.title, thesis.description, thesis.scope, thesis.objectives, thesis.expected_result
            )
            similarity = content_similarity(incoming, existing)
            if similarity >= 0.85:
                candidates.append((thesis, similarity))
        return candidates

    def create(self, thesis: Thesis) -> Thesis:
        self.db.add(thesis)
        self.db.flush()
        return thesis

    def list_paginated(self, page: int, page_size: int, semester: str | None = None, program: str | None = None, domain: str | None = None, technology: str | None = None, keyword: str | None = None):
        query = self.db.query(Thesis).filter(Thesis.is_deleted.is_(False))
        if semester:
            query = query.filter(Thesis.semester == semester)
        if program:
            query = query.filter(Thesis.program == program)
        if keyword:
            query = query.filter(or_(Thesis.title.ilike(f"%{keyword}%"), Thesis.description.ilike(f"%{keyword}%")))
        if domain:
            query = query.join(Thesis.domains).filter_by(name=domain)
        if technology:
            query = query.join(Thesis.technologies).filter_by(name=technology)
        return query.order_by(Thesis.thesis_id.desc()).offset((page - 1) * page_size).limit(page_size).all()

    def get_detail(self, thesis_id: int) -> Optional[Thesis]:
        return (
            self.db.query(Thesis)
            .options(
                joinedload(Thesis.domains),
                joinedload(Thesis.semantics),
                joinedload(Thesis.structures),
                joinedload(Thesis.lexical_tags),
                joinedload(Thesis.technologies),
            )
            .filter(Thesis.thesis_id == thesis_id, Thesis.is_deleted.is_(False))
            .first()
        )
