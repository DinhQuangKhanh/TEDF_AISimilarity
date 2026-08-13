"""Load the recent-semester capstone corpus (Spring 2026 + Summer 2026) from the JSON files as
full-content, thesis-like objects — no database required (P4).

Used by the demo analyze flow so that a fully-filled topic is compared, field-for-field, against
the two most recent semesters' topics (which also carry all six fields).
"""

from __future__ import annotations

import json
import os

from app.services.preprocessing import concept_names, preprocess

_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data"
)
_FILES = ("capstone_SP26.json", "capstone_SU26.json")


class _Tag:
    __slots__ = ("name",)

    def __init__(self, name: str):
        self.name = name


class CorpusTopic:
    """Full-content thesis-like object the scorer can read (title + 5 fields + tags), no DB row."""

    def __init__(self, detail: dict):
        self.title = detail.get("titleEn")
        self.description = detail.get("description")
        self.scope = detail.get("scope")
        self.objectives = detail.get("objective")
        self.expected_result = detail.get("expectedResult")
        self.semester = detail.get("semester")
        self.technologies = [_Tag(t) for t in _split_tech(detail.get("technology"))]
        # Domain / structure tags from Module 1 on the title (same convention as seed_thesis.py / P1.7).
        module1 = preprocess(self.title or "")
        self.domains = [_Tag(name) for name in concept_names(module1.domains)]
        self.structures = [_Tag(name) for name in concept_names(module1.methods | module1.tasks)]


def _split_tech(value: str | None) -> list[str]:
    return [t.strip() for t in (value or "").split(",") if t.strip()]


def load_recent_capstone_corpus() -> list[CorpusTopic]:
    """Every topic in capstone_SP26.json + capstone_SU26.json as a CorpusTopic (~53 topics)."""
    topics: list[CorpusTopic] = []
    for fname in _FILES:
        with open(os.path.join(_DATA_DIR, fname), encoding="utf-8") as handle:
            data = json.load(handle)
        topics.extend(CorpusTopic(detail) for detail in data.values() if detail.get("titleEn"))
    return topics
