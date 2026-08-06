from uuid import UUID

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas.import_schema import ApiResponse
from app.services.similarity_service import SimilarityService

router = APIRouter(prefix="/api/v1/similarity", tags=["similarity"])


@router.post("/run-new", response_model=ApiResponse)
def run_similarity_for_new(thesis_ids: list[UUID], db: Session = Depends(get_db)):
    service = SimilarityService(db)
    service.run_for_new(thesis_ids)
    db.commit()
    return ApiResponse(
        success=True,
        message="Similarity completed",
        data={"processed_ids": thesis_ids, "similarities": service.results_for(thesis_ids)},
    )
