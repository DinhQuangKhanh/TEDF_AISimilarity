from sqlalchemy.orm import Session

from app.repositories.audit_repository import AuditLogRepository


class AuditService:
    def __init__(self, db: Session):
        self.repo = AuditLogRepository(db)

    def log(self, action: str, table_name: str, record_id: int | None = None, detail: str | None = None):
        return self.repo.create(action=action, table_name=table_name, record_id=record_id, detail=detail)
