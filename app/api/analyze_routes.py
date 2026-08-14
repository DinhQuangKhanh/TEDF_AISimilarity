"""Demo endpoints: a dry-run pipeline analyzer + the static demo UI page.

`POST /api/v1/similarity/analyze` runs the full DASSF pipeline on a typed topic (no persistence)
and returns the step-by-step trace for the UI. `GET /demo` serves the single-file demo page —
served from the API itself so it is same-origin and can call the endpoint without CORS.
"""

from __future__ import annotations

import os

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from app.schemas.import_schema import ApiResponse
from app.services.analyze_service import analyze_topic
from app.services.corpus_loader import load_recent_capstone_corpus
from app.services.lexical import build_lexical_scorer
from app.services.semantic_encoder import backend_name

router = APIRouter(tags=["demo"])

_DEMO_HTML = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static", "demo.html"
)


class AnalyzeRequest(BaseModel):
    title: str
    description: str | None = None
    scope: str | None = None
    objectives: str | None = None
    expected_result: str | None = None
    technologies: list[str] = []


class _Tag:
    __slots__ = ("name",)

    def __init__(self, name: str):
        self.name = name


class _QueryTopic:
    """Transient, unsaved thesis-like object the scorer can read (no DB row created)."""

    def __init__(self, req: AnalyzeRequest):
        self.title = req.title
        self.description = req.description
        self.scope = req.scope
        self.objectives = req.objectives
        self.expected_result = req.expected_result
        self.technologies = [_Tag(t) for t in req.technologies]
        self.structures = []
        self.domains = []


@router.post("/api/v1/similarity/analyze", response_model=ApiResponse)
def analyze(req: AnalyzeRequest):
    # Compare against the two most-recent semesters (Spring 2026 + Summer 2026), loaded full-content
    # from the capstone JSON — no database needed for the demo (P4).
    corpus = load_recent_capstone_corpus()
    lexical_model = build_lexical_scorer(corpus) if corpus else None
    result = analyze_topic(_QueryTopic(req), corpus, lexical_model, semantic_backend=backend_name())
    return ApiResponse(success=True, message="Analyzed", data=result)


@router.get("/demo", include_in_schema=False)
def demo_page():
    if not os.path.exists(_DEMO_HTML):
        raise HTTPException(status_code=404, detail="Demo page not found")
    with open(_DEMO_HTML, encoding="utf-8") as handle:
        return HTMLResponse(handle.read())
