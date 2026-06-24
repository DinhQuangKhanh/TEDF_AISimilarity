from app.models.thesis import Thesis
from app.utils.duplicate_checker import is_duplicate


def test_is_duplicate_false(db_session):
    assert not is_duplicate(db_session, "Title A", None, "2025 Spring", "SE")


def test_is_duplicate_true(db_session):
    db_session.add(Thesis(title_en="Title A", semester="2025 Spring", program="SE"))
    db_session.commit()
    assert is_duplicate(db_session, "Title A", None, "2025 Spring", "SE")
