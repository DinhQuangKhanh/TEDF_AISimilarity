import hashlib

from sqlalchemy.orm import Session

from app.models.thesis import Thesis
from app.utils.text_cleaner import normalize_key, tokenize

# Các trường nội dung dùng để xác định trùng lặp.
_CONTENT_FIELDS = ("title", "description", "scope", "objectives", "expected_result")
_FIELD_SEPARATOR = "\x1f"


def content_hash(
    title: str | None,
    description: str | None,
    scope: str | None,
    objectives: str | None,
    expected_result: str | None,
    semester: str | None,
    program: str | None,
) -> str:
    """Băm 5 trường nội dung + semester + program để so trùng CHÍNH XÁC.

    Dùng normalize_key nên bỏ qua khác biệt hoa/thường và khoảng trắng.
    """
    parts = [
        normalize_key(value)
        for value in (title, description, scope, objectives, expected_result, semester, program)
    ]
    joined = _FIELD_SEPARATOR.join(parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def combined_content(
    title: str | None,
    description: str | None,
    scope: str | None,
    objectives: str | None,
    expected_result: str | None,
) -> str:
    """Ghép 5 trường nội dung (đã chuẩn hoá) để so GẦN trùng."""
    parts = [normalize_key(value) for value in (title, description, scope, objectives, expected_result)]
    return " ".join(part for part in parts if part)


def is_duplicate(
    db: Session,
    title: str | None,
    description: str | None,
    scope: str | None,
    objectives: str | None,
    expected_result: str | None,
    semester: str | None,
    program: str | None,
) -> bool:
    """Trùng chính xác khi băm nội dung 5 trường (+ semester + program) đã tồn tại."""
    digest = content_hash(title, description, scope, objectives, expected_result, semester, program)
    return (
        db.query(Thesis)
        .filter(Thesis.is_deleted.is_(False), Thesis.content_hash == digest)
        .first()
        is not None
    )


def content_similarity(left: str | None, right: str | None) -> float:
    """Độ giống nội dung theo Jaccard trên token.

    Bền với văn bản dài và thứ tự từ (khác SequenceMatcher ở mức ký tự).
    """
    left_tokens = tokenize(left)
    right_tokens = tokenize(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)
