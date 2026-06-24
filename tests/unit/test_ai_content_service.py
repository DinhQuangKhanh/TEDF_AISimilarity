from app.services.ai_content_service import AIContentService


def test_ai_content_service_fallback_generation():
    service = AIContentService()
    payload = service.generate_description_and_scope(title_vi=None, title_en="Hospital Management System")
    assert payload["description"] is not None
    assert payload["scope"] is not None
    assert payload["source"] in {"inferred_by_ai_fallback", "inferred_by_ai"}


def test_ai_content_service_empty_title():
    service = AIContentService()
    payload = service.generate_description_and_scope(title_vi=None, title_en=None)
    assert payload["description"] is None
    assert payload["scope"] is None
