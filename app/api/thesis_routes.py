from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.core.config import DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE
from app.database import get_db
from app.repositories.similarity_repository import SimilarityRepository
from app.repositories.thesis_repository import ThesisRepository
from app.schemas.import_schema import ApiResponse
from app.schemas.thesis_schema import ThesisCreateRequest, ThesisDetail, ThesisListItem
from app.services.explanation import build_explanation
from app.services.similarity_service import SimilarityService
from app.services.thesis_service import ThesisService
from app.services.translation_service import translate_to_vietnamese
from app.utils.score_calculator import calculate_scores

router = APIRouter(prefix="/api/v1/theses", tags=["theses"])


def _detail_payload(thesis) -> dict:
    return ThesisDetail(
        thesis_id=thesis.thesis_id,
        semester=thesis.semester,
        program=thesis.program,
        title=thesis.title,
        description=thesis.description,
        scope=thesis.scope,
        objectives=thesis.objectives,
        expected_result=thesis.expected_result,
        domains=[item.name for item in thesis.domains],
        semantic_categories=[item.name for item in thesis.semantics],
        structure_types=[item.name for item in thesis.structures],
        lexical_tags=[item.name for item in thesis.lexical_tags],
        technologies=[item.name for item in thesis.technologies],
    ).model_dump()


@router.post("", response_model=ApiResponse, status_code=201)
def create_thesis(payload: ThesisCreateRequest, db: Session = Depends(get_db)):
    """Add one new topic, then score it against every existing topic."""
    try:
        thesis_id = ThesisService(db).create_thesis(payload.to_raw_data(), payload.thesis_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail={"success": False, "message": "Validation failed", "errors": [str(exc)]},
        )
    if thesis_id is None:
        raise HTTPException(
            status_code=409,
            detail={
                "success": False,
                "message": "Duplicate topic",
                "errors": ["An identical topic already exists for this semester and program"],
            },
        )
    thesis = ThesisRepository(db).get_detail(thesis_id)
    similarities = SimilarityService(db).results_for([thesis_id])
    return ApiResponse(
        success=True,
        message="Thesis created",
        data={
            "thesis": _detail_payload(thesis),
            "needs_review": thesis.needs_review,
            "similarities": similarities,
        },
    )


@router.get("", response_model=ApiResponse)
def list_theses(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE),
    semester: str | None = None,
    program: str | None = None,
    domain: str | None = None,
    technology: str | None = None,
    keyword: str | None = None,
    db: Session = Depends(get_db),
):
    theses = ThesisRepository(db).list_paginated(page, page_size, semester, program, domain, technology, keyword)
    items = [ThesisListItem.model_validate(thesis).model_dump() for thesis in theses]
    return ApiResponse(success=True, message="Thesis list fetched", data={"items": items, "page": page, "page_size": page_size})


@router.get("/{thesis_id}", response_model=ApiResponse)
def get_thesis_detail(thesis_id: UUID, db: Session = Depends(get_db)):
    thesis = ThesisRepository(db).get_detail(thesis_id)
    if not thesis:
        raise HTTPException(status_code=404, detail={"success": False, "message": "Not found", "errors": ["Thesis not found"]})
    return ApiResponse(success=True, message="Thesis detail fetched", data=_detail_payload(thesis))


@router.get("/{thesis_id}/similarities", response_model=ApiResponse)
def get_thesis_similarities(thesis_id: UUID, db: Session = Depends(get_db)):
    items = SimilarityService(db).results_for([thesis_id])
    return ApiResponse(success=True, message="Similarity results fetched", data={"items": items})


@router.get("/{thesis_id}/explain/{other_id}", response_model=ApiResponse)
def explain_pair(thesis_id: UUID, other_id: UUID, db: Session = Depends(get_db)):
    """Committee-facing explanation for one topic vs a matched topic (paper §V-F): the composite
    score, the per-dimension breakdown, the specific overlapping SEDO concepts, and a concrete
    revision suggestion. Reuses the stored similarity scores when available, else computes fresh."""
    a = ThesisRepository(db).get_detail(thesis_id)
    b = ThesisRepository(db).get_detail(other_id)
    if not a or not b:
        raise HTTPException(status_code=404, detail={"success": False, "message": "Not found", "errors": ["Thesis not found"]})

    row = SimilarityRepository(db).get_pair(thesis_id, other_id)
    if row is not None:
        scores = {
            "semantic_score": row.semantic_score,
            "lexical_score": row.lexical_score,
            "structure_score": row.structure_score,
            "domain_score": row.domain_score,
            "overall_score": row.overall_score,
            "level": row.level,
        }
    else:
        scores = calculate_scores(a, b)

    report = build_explanation(a, b, scores)
    report["thesis_id"] = str(a.thesis_id)
    report["other_thesis_id"] = str(b.thesis_id)
    return ApiResponse(success=True, message="Explanation generated", data=report)


@router.get("/{thesis_id}/translate", response_model=ApiResponse)
def translate_thesis(thesis_id: UUID, db: Session = Depends(get_db)):
    """Returns the topic's content translated to Vietnamese for the side-by-side comparison view."""
    thesis = ThesisRepository(db).get_detail(thesis_id)
    if not thesis:
        raise HTTPException(status_code=404, detail={"success": False, "message": "Not found", "errors": ["Thesis not found"]})

    translated = translate_to_vietnamese(
        {
            "title": thesis.title,
            "description": thesis.description,
            "scope": thesis.scope,
            "objectives": thesis.objectives,
            "expected_result": thesis.expected_result,
        }
    )
    return ApiResponse(
        success=True,
        message="Translation completed",
        data={
            "thesis_id": str(thesis.thesis_id),
            "title": translated["title"],
            "description": translated["description"],
            "scope": translated["scope"],
            "objectives": translated["objectives"],
            "expected_result": translated["expected_result"],
            "technologies": [item.name for item in thesis.technologies],
            "translated": translated["translated"],
        },
    )
