from app.models.thesis import Thesis
from app.utils.duplicate_checker import content_hash, is_duplicate

FIELDS = dict(
    title="Title A",
    description="A description",
    scope="A scope",
    objectives="Some objectives",
    expected_result="Some expected result",
    semester="2025 Spring",
    program="SE",
)


def test_is_duplicate_false(db_session):
    assert not is_duplicate(db_session, **FIELDS)


def test_is_duplicate_true(db_session):
    db_session.add(
        Thesis(
            title=FIELDS["title"],
            description=FIELDS["description"],
            scope=FIELDS["scope"],
            objectives=FIELDS["objectives"],
            expected_result=FIELDS["expected_result"],
            semester=FIELDS["semester"],
            program=FIELDS["program"],
            content_hash=content_hash(**FIELDS),
        )
    )
    db_session.commit()
    assert is_duplicate(db_session, **FIELDS)


def test_is_duplicate_ignores_other_content(db_session):
    db_session.add(
        Thesis(
            title=FIELDS["title"],
            description="different description",
            scope=FIELDS["scope"],
            objectives=FIELDS["objectives"],
            expected_result=FIELDS["expected_result"],
            semester=FIELDS["semester"],
            program=FIELDS["program"],
            content_hash=content_hash(
                title=FIELDS["title"],
                description="different description",
                scope=FIELDS["scope"],
                objectives=FIELDS["objectives"],
                expected_result=FIELDS["expected_result"],
                semester=FIELDS["semester"],
                program=FIELDS["program"],
            ),
        )
    )
    db_session.commit()
    # Cùng title nhưng khác description -> KHÔNG coi là trùng chính xác.
    assert not is_duplicate(db_session, **FIELDS)
