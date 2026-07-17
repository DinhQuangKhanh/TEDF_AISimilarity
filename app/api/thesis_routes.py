import json

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.core.config import DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE
from app.database import get_db
from app.repositories.similarity_repository import SimilarityRepository
from app.repositories.thesis_repository import ThesisRepository
from app.schemas.import_schema import ApiResponse
from app.schemas.thesis_schema import ThesisDetail, ThesisListItem

router = APIRouter(prefix="/api/v1/theses", tags=["theses"])


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
def get_thesis_detail(thesis_id: int, db: Session = Depends(get_db)):
    thesis = ThesisRepository(db).get_detail(thesis_id)
    if not thesis:
        raise HTTPException(status_code=404, detail={"success": False, "message": "Not found", "errors": ["Thesis not found"]})
    payload = ThesisDetail(
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
    )
    return ApiResponse(success=True, message="Thesis detail fetched", data=payload.model_dump())


@router.get("/{thesis_id}/similarities", response_model=ApiResponse)
def get_thesis_similarities(thesis_id: int, db: Session = Depends(get_db)):
    rows = SimilarityRepository(db).list_for_thesis(thesis_id)
    items = []
    for row in rows:
        items.append(
            {
                "sim_id": row.sim_id,
                "thesis_a_id": row.thesis_a_id,
                "thesis_b_id": row.thesis_b_id,
                "semantic_score": row.semantic_score,
                "lexical_score": row.lexical_score,
                "structure_score": row.structure_score,
                "domain_score": row.domain_score,
                "overall_score": row.overall_score,
                "level": row.level,
                "reason": json.loads(row.reason) if row.reason else [],
            }
        )
    return ApiResponse(success=True, message="Similarity results fetched", data={"items": items})
