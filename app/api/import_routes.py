import os

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from sqlalchemy.orm import Session

from app.core.config import ALLOWED_EXTENSIONS, MAX_UPLOAD_MB
from app.database import get_db
from app.schemas.import_schema import ApiResponse
from app.services.excel_reader_service import ExcelReaderService
from app.services.thesis_service import ThesisService

router = APIRouter(prefix="/api/v1/import", tags=["import"])


def _validate_file(file: UploadFile, content: bytes) -> None:
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail={"success": False, "message": "Validation failed", "errors": ["Unsupported file type"]})
    if len(content) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=400, detail={"success": False, "message": "Validation failed", "errors": ["File too large"]})


@router.post("/excel", response_model=ApiResponse)
async def import_excel(
    file: UploadFile = File(...),
    semester: str | None = Query(default=None),
    program: str | None = Query(default=None),
    db: Session = Depends(get_db),
):
    content = await file.read()
    _validate_file(file, content)
    rows = ExcelReaderService().parse(content)
    if not rows:
        raise HTTPException(status_code=400, detail={"success": False, "message": "Validation failed", "errors": ["No valid data found in Excel file"]})
    new_ids, errors = ThesisService(db).import_rows(rows, semester, program)
    return ApiResponse(success=True, message="Import completed", data={"inserted": len(new_ids), "new_ids": new_ids, "errors": errors})
