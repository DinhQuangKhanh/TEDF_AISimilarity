import os
from typing import Optional

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
        semester: Optional[str] = None,
        program: Optional[str] = None,
    ) -> NormalizedRow:
        data = raw_row["data"]
        title_en = clean_text(data.get("title_en"))
        title_vn = clean_text(data.get("title_vn"))
        description = clean_text(data.get("description"))
        scope = clean_text(data.get("scope"))
        title = title_en or title_vn
        notes = [f"Imported from sheet {raw_row['sheet_name']} row {raw_row['row_number']}"]

        if not description and title:
            description = self._generate_description(title_en=title_en, title_vn=title_vn)
            notes.append("description_inferred_by_ai")

        if not scope and title:
            scope = self._generate_scope(title_en=title_en, title_vn=title_vn, description=description)
            notes.append("scope_inferred_by_ai")

        domains = normalize_list([data.get("domains")] if data.get("domains") else [])
        technologies = normalize_list(
            [item.strip() for item in (data.get("technologies") or "").split(",") if item.strip()]
        )
        lexical = normalize_list([*(title_en or "").split(), *(title_vn or "").split()])[:10]
        semantic = normalize_list(domains or ["General"])
        structures = normalize_list(
            ["Web Application"]
            if any(technology in technologies for technology in ["React", "Node.js", "FastAPI"])
            else ["Software Project"]
        )

        return NormalizedRow(
            semester=clean_text(data.get("semester")) or semester,
            program=clean_text(data.get("program")) or program,
            title_en=title_en,
            title_vn=title_vn,
            description=description,
            scope=scope,
            domains=domains,
            semantic_categories=semantic,
            structure_types=structures,
            lexical_tags=lexical,
            technologies=technologies,
            notes="; ".join(notes),
        )

    def _generate_description(self, title_en: str | None, title_vn: str | None) -> str:
        title = title_en or title_vn or "the thesis project"
        if OPENAI_API_KEY:
            try:
                return _call_openai(
                    "Generate a concise thesis/project description from the title. "
                    "Return plain text only, no markdown, max 80 words.\n"
                    f"English title: {title_en or ''}\n"
                    f"Vietnamese title: {title_vn or ''}"
                )
            except Exception:
                pass
        return (
            f"This thesis focuses on designing and implementing {title.lower()}, "
            "covering the main user workflows, data management, and system evaluation."
        )

    def _generate_scope(self, title_en: str | None, title_vn: str | None, description: str | None) -> str:
        title = title_en or title_vn or "the thesis project"
        if OPENAI_API_KEY:
            try:
                return _call_openai(
                    "Generate a concise project scope from the title and description. "
                    "Return plain text only, no markdown, max 60 words.\n"
                    f"English title: {title_en or ''}\n"
                    f"Vietnamese title: {title_vn or ''}\n"
                    f"Description: {description or ''}"
                )
            except Exception:
                pass
        return (
            f"The scope includes requirement analysis, core feature implementation, testing, "
            f"and deployment preparation for {title.lower()}."
        )
