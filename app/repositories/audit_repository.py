from typing import Optional

from sqlalchemy.orm import Session

from app.models.audit_log import AuditLog


class AuditLogRepository:
    def __init__(self, db: Session):
        self.db = db

    def create(self, action: str, table_name: str, record_id: Optional[int] = None, detail: Optional[str] = None) -> AuditLog:
        audit = AuditLog(action=action, table_name=table_name, record_id=record_id, detail=detail)
        self.db.add(audit)
        self.db.flush()
        return audit
