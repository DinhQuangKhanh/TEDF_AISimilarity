import os

import httpx

from app.schemas.import_schema import NormalizedRow
from app.utils.text_cleaner import clean_text, normalize_list

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def _call_openai(prompt: str) -> str:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}",
    }
    payload = {
        "model": OPENAI_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
    }
    response = httpx.post(
        "https://api.openai.com/v1/chat/completions",
        json=payload,
        headers=headers,
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


class LLMNormalizerService:
    def normalize(
        self,
        raw_row: dict,
        semester: str | None = None,
        program: str | None = None,
    ) -> NormalizedRow:
        data = raw_row["data"]
        title = clean_text(data.get("title"))
        description = clean_text(data.get("description"))
        scope = clean_text(data.get("scope"))
        objectives = clean_text(data.get("objectives"))
        expected_result = clean_text(data.get("expected_result"))

        # Thiếu trường nội dung nào & có title -> để AI (hoặc heuristic) suy ra.
        if title:
            if not description:
                description = self._generate_description(title)
            if not scope:
                scope = self._generate_scope(title, description)
            if not objectives:
                objectives = self._generate_objectives(title, description)
            if not expected_result:
                expected_result = self._generate_expected_result(title, description)

        domains = normalize_list([data.get("domains")] if data.get("domains") else [])
        technologies = normalize_list(
            [item.strip() for item in (data.get("technologies") or "").split(",") if item.strip()]
        )
        lexical = normalize_list((title or "").split())[:10]
        semantic = normalize_list(domains or ["General"])
        structures = normalize_list(
            ["Web Application"]
            if any(technology in technologies for technology in ["React", "Node.js", "FastAPI"])
            else ["Software Project"]
        )

        return NormalizedRow(
            semester=clean_text(data.get("semester")) or semester,
            program=clean_text(data.get("program")) or program,
            title=title,
            description=description,
            scope=scope,
            objectives=objectives,
            expected_result=expected_result,
            domains=domains,
            semantic_categories=semantic,
            structure_types=structures,
            lexical_tags=lexical,
            technologies=technologies,
        )

    def _generate_description(self, title: str) -> str:
        if OPENAI_API_KEY:
            try:
                return _call_openai(
                    "Generate a concise thesis/project description from the title. "
                    "Return plain text only, no markdown, max 80 words.\n"
                    f"Title: {title}"
                )
            except Exception:
                pass
        return (
            f"This thesis focuses on designing and implementing {title.lower()}, "
            "covering the main user workflows, data management, and system evaluation."
        )

    def _generate_scope(self, title: str, description: str | None) -> str:
        if OPENAI_API_KEY:
            try:
                return _call_openai(
                    "Generate a concise project scope from the title and description. "
                    "Return plain text only, no markdown, max 60 words.\n"
                    f"Title: {title}\n"
                    f"Description: {description or ''}"
                )
            except Exception:
                pass
        return (
            "The scope includes requirement analysis, core feature implementation, testing, "
            f"and deployment preparation for {title.lower()}."
        )

    def _generate_objectives(self, title: str, description: str | None) -> str:
        if OPENAI_API_KEY:
            try:
                return _call_openai(
                    "Generate concise objectives (2-4 short goals) for the thesis/project "
                    "from the title and description. Return plain text only, no markdown, max 60 words.\n"
                    f"Title: {title}\n"
                    f"Description: {description or ''}"
                )
            except Exception:
                pass
        return (
            f"The main objectives are to analyze requirements, build the core features of {title.lower()}, "
            "and evaluate the system against the expected outcomes."
        )

    def _generate_expected_result(self, title: str, description: str | None) -> str:
        if OPENAI_API_KEY:
            try:
                return _call_openai(
                    "Generate the expected result/outcome for the thesis/project "
                    "from the title and description. Return plain text only, no markdown, max 60 words.\n"
                    f"Title: {title}\n"
                    f"Description: {description or ''}"
                )
            except Exception:
                pass
        return (
            f"The expected result is a working implementation of {title.lower()} that meets the defined "
            "requirements and passes functional evaluation."
        )
