"""Tests for the importer normalizer — no fabricated content, tags from real text (P1.6/P1.7)."""

from app.services.llm_normalizer_service import LLMNormalizerService


def _row(**data) -> dict:
    return {"sheet_name": "test", "row_number": 1, "data": data}


def test_missing_fields_are_not_fabricated_when_fill_disabled():
    result = LLMNormalizerService().normalize(
        _row(title="Hotel Management System with React"),
        fill_missing=False,
    )
    assert result.title == "Hotel Management System with React"
    assert result.description is None
    assert result.scope is None
    assert result.objectives is None
    assert result.expected_result is None


def test_tags_are_derived_from_real_title():
    result = LLMNormalizerService().normalize(
        _row(title="Hotel Management System with React and Node.js"),
        fill_missing=False,
    )
    assert "React" in result.technologies
    assert "Hotel" in result.domains
    # never fabricated, but classification is still populated from the text
    assert result.description is None


def test_provided_content_is_preserved():
    result = LLMNormalizerService().normalize(
        _row(title="Pharmacy System", description="A real description written by the author."),
        fill_missing=False,
    )
    assert result.description == "A real description written by the author."
    assert "Pharmacy" in result.domains


def test_csv_tags_are_merged_even_if_not_in_ontology():
    result = LLMNormalizerService().normalize(
        _row(title="Realtime bus tracker", technologies="Kafka, React"),
        fill_missing=False,
    )
    assert "Kafka" in result.technologies      # preserved from CSV even though SEDO lacks it
    assert "React" in result.technologies


def test_opt_in_fill_uses_heuristic_without_api_key(monkeypatch):
    # With fill enabled and no OpenAI key, the heuristic fills the fields (non-null) — opt-in only.
    monkeypatch.setattr("app.services.llm_normalizer_service.OPENAI_API_KEY", None)
    result = LLMNormalizerService().normalize(
        _row(title="Library Management System"),
        fill_missing=True,
    )
    assert result.description is not None
    assert "library management system" in result.description.lower()


def test_title_extracts_domain():
    result = LLMNormalizerService().normalize(
        _row(title="Hotel Management System"),
        fill_missing=False,
    )
    assert "Hotel" in result.domains
