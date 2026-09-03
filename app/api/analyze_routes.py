"""Demo endpoints: a dry-run pipeline analyzer + the static demo UI page.

`POST /api/v1/similarity/analyze` runs the full DASSF pipeline on a typed topic (no persistence)
and returns the step-by-step trace for the UI. `GET /api/v1/similarity/model-info` reports the
deployed MDDM weights, where they came from, and — when `tools/trace_tuning.py` has been run — the
recorded grid-search log so the page can replay how those weights were learned. `GET /demo` serves
the single-file demo page — served from the API itself so it is same-origin and needs no CORS.
"""

from __future__ import annotations

import json
import os
from importlib.util import find_spec

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from app.core.config import MDDM_WEIGHTS, MDDM_WEIGHTS_SOURCE
from app.schemas.import_schema import ApiResponse
from app.services.analyze_service import analyze_topic
from app.services.corpus_loader import load_recent_capstone_corpus
from app.services.lexical import build_lexical_scorer
from app.services.semantic_encoder import backend_name
from app.utils.score_calculator import LEVEL_THRESHOLDS, TAU_DOM, TAU_STR

router = APIRouter(tags=["demo"])

_APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEMO_HTML = os.path.join(_APP_DIR, "static", "demo.html")
_TUNING_TRACE = os.path.join(os.path.dirname(_APP_DIR), "data", "tuning_trace.json")


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


class TopicFields(BaseModel):
    title: str | None = None
    description: str | None = None
    scope: str | None = None
    objectives: str | None = None
    expected_result: str | None = None
    technologies: list[str] = []


class ExplainRequest(BaseModel):
    query: TopicFields
    match: TopicFields


def _topic_from_fields(fields: TopicFields) -> _QueryTopic:
    return _QueryTopic(AnalyzeRequest(
        title=fields.title or "",
        description=fields.description,
        scope=fields.scope,
        objectives=fields.objectives,
        expected_result=fields.expected_result,
        technologies=fields.technologies,
    ))


@router.post("/api/v1/similarity/explain", response_model=ApiResponse)
def explain_route(req: ExplainRequest):
    # Per-field "explain duplication": recomputes the SAME highlights as /analyze for this one pair,
    # then phrases each field's overlap (LLM if available, else a grounded template). No persistence.
    from app.services.explain_service import explain

    fields = explain(_topic_from_fields(req.query), _topic_from_fields(req.match))
    return ApiResponse(success=True, message="Explained", data={"fields": fields})


@router.get("/api/v1/similarity/model-info", response_model=ApiResponse)
def model_info():
    """The four MDDM weights currently in force, the decision cut-offs they feed, and — when
    ``tools/trace_tuning.py`` has been run — the full grid-search log that produced them."""
    tuning = None
    if os.path.exists(_TUNING_TRACE):
        try:
            with open(_TUNING_TRACE, encoding="utf-8") as handle:
                tuning = json.load(handle)
        except (OSError, ValueError):  # unreadable / half-written file ⇒ just omit the log
            tuning = None
    return ApiResponse(success=True, message="Model info", data={
        "weights": MDDM_WEIGHTS,
        "weightsSource": MDDM_WEIGHTS_SOURCE,
        "levelThresholds": [{"min": lo, "level": lv} for lo, lv in LEVEL_THRESHOLDS],
        "structuralRule": {"tauStr": TAU_STR, "tauDom": TAU_DOM},
        "backends": {
            "semantic": backend_name(),
            "lexical": "tfidf-cosine" if find_spec("sklearn") else "weighted-jaccard-fallback",
        },
        "tuning": tuning,
        "tuningHint": "python tools/trace_tuning.py" if tuning is None else None,
    })


@router.get("/demo", include_in_schema=False)
def demo_page():
    if not os.path.exists(_DEMO_HTML):
        raise HTTPException(status_code=404, detail="Demo page not found")
    with open(_DEMO_HTML, encoding="utf-8") as handle:
        return HTMLResponse(handle.read())
