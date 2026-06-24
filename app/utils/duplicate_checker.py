from difflib import SequenceMatcher

from sqlalchemy.orm import Session

from app.models.thesis import Thesis
from app.utils.text_cleaner import normalize_key


def is_duplicate(
    db: Session,
    title_en: str | None,
    title_vn: str | None,
    semester: str | None,
    program: str | None,
) -> bool:
    query = db.query(Thesis).filter(Thesis.is_deleted.is_(False))
    if title_en:
        query = query.filter(Thesis.title_en == title_en)
    if title_vn:
        query = query.filter(Thesis.title_vn == title_vn)
    if semester:
        query = query.filter(Thesis.semester == semester)
    if program:
        query = query.filter(Thesis.program == program)
    return query.first() is not None


def title_similarity(left: str | None, right: str | None) -> float:
    return SequenceMatcher(None, normalize_key(left), normalize_key(right)).ratio()
