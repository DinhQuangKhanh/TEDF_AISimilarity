import json
import os
from typing import Any

from app.core.logging import logger
from app.utils.text_cleaner import clean_text


class AIContentService:
    def __init__(self) -> None:
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

    def generate_description_and_scope(self, title_vi: str | None, title_en: str | None) -> dict[str, Any]:
        title = clean_text(title_en) or clean_text(title_vi)
        if not title:
            return {
                "description": None,
                "scope": None,
                "source": "unavailable",
                "confidence": 0.0,
            }

        generated = self._generate_with_openai(title_vi=title_vi, title_en=title_en)
        if generated:
            return generated
        return self._generate_with_heuristic(title_vi=title_vi, title_en=title_en)

    def _generate_with_openai(self, title_vi: str | None, title_en: str | None) -> dict[str, Any] | None:
        if not self.openai_api_key:
            return None
        try:
            from openai import OpenAI

            client = OpenAI(api_key=self.openai_api_key)
            prompt = (
                "You normalize thesis metadata. Return JSON only with keys "
                "description, scope, confidence. Infer concise academic project content from the title. "
                "If uncertain, keep the field null.\n"
                f"Vietnamese title: {title_vi or ''}\n"
                f"English title: {title_en or ''}"
            )
            response = client.responses.create(
                model=self.openai_model,
                input=prompt,
                max_output_tokens=250,
            )
            text = getattr(response, "output_text", None)
            if not text:
                return None
            payload = json.loads(text)
            return {
                "description": clean_text(payload.get("description")),
                "scope": clean_text(payload.get("scope")),
                "source": "inferred_by_ai",
                "confidence": float(payload.get("confidence", 0.7)),
            }
        except Exception:
            logger.exception("Failed to generate description/scope with OpenAI; falling back to heuristic")
            return None

    def _generate_with_heuristic(self, title_vi: str | None, title_en: str | None) -> dict[str, Any]:
        title = clean_text(title_en) or clean_text(title_vi) or "software project"
        description = (
            f"This thesis focuses on designing and implementing {title.lower()}, "
            "including core workflows, user interaction, data processing, and system evaluation."
        )
        scope = (
            f"The project scope covers requirement analysis, architecture design, implementation, testing, "
            f"and deployment preparation for {title.lower()}."
        )
        return {
            "description": description,
            "scope": scope,
            "source": "inferred_by_ai_fallback",
            "confidence": 0.45,
        }
