import uuid

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.models.classification import Domain, LexicalTag, SemanticCategory, StructureType, Tech
from app.models.thesis import Thesis
from app.repositories.classification_repository import ClassificationRepository
from app.repositories.thesis_repository import ThesisRepository
from app.services.audit_service import AuditService
from app.services.llm_normalizer_service import LLMNormalizerService
from app.services.schema_mapper_service import SchemaMapperService
from app.services.similarity_service import SimilarityService
from app.utils.duplicate_checker import content_hash, is_duplicate


class ThesisService:
    def __init__(self, db: Session):
        self.db = db
        self.repo = ThesisRepository(db)
        self.class_repo = ClassificationRepository(db)
        self.audit = AuditService(db)
        self.normalizer = LLMNormalizerService()
        self.mapper = SchemaMapperService()
        self.similarity_service = SimilarityService(db)

    def _attach_classifications(self, thesis: Thesis, thesis_data) -> None:
        for name in thesis_data.domains:
            thesis.domains.append(self.class_repo.get_or_create(Domain, name))
        for name in thesis_data.semantic_categories:
            thesis.semantics.append(self.class_repo.get_or_create(SemanticCategory, name))
        for name in thesis_data.structure_types:
            thesis.structures.append(self.class_repo.get_or_create(StructureType, name))
        for name in thesis_data.lexical_tags:
            thesis.lexical_tags.append(self.class_repo.get_or_create(LexicalTag, name))
        for name in thesis_data.technologies:
            thesis.technologies.append(self.class_repo.get_or_create(Tech, name))

    def _persist_row(
        self, raw: dict, semester: str | None, program: str | None, thesis_id: uuid.UUID | None = None
    ) -> uuid.UUID | None:
        """Normalize, map, dedupe and persist one raw row.

        ``thesis_id`` lets the caller pin the id (the web project's UUID); when omitted a
        fresh UUID is generated (Excel imports). Returns the id, or None on exact duplicate.
        """
        normalized = self.normalizer.normalize(raw, semester, program)
        self.audit.log("NORMALIZE_WITH_LLM", "thesis", None, f"Normalized row {raw['row_number']}")
        thesis_data = self.mapper.map(normalized)
        if not thesis_data.title:
            raise ValueError("A title is required")
        # Both duplicate checks read all five content fields (paper Sect. 3.1).
        if is_duplicate(
            self.db,
            thesis_data.title,
            thesis_data.description,
            thesis_data.scope,
            thesis_data.objectives,
            thesis_data.expected_result,
            thesis_data.semester,
            thesis_data.program,
        ):
            self.audit.log("SKIP_DUPLICATE", "thesis", None, f"Duplicate row {raw['row_number']}")
            return None
        candidates = self.repo.find_near_duplicate_candidates(
            thesis_data.title,
            thesis_data.description,
            thesis_data.scope,
            thesis_data.objectives,
            thesis_data.expected_result,
            thesis_data.semester,
            thesis_data.program,
        )
        thesis = Thesis(
            thesis_id=thesis_id or uuid.uuid4(),
            semester=thesis_data.semester,
            program=thesis_data.program,
            title=thesis_data.title,
            description=thesis_data.description,
            scope=thesis_data.scope,
            objectives=thesis_data.objectives,
            expected_result=thesis_data.expected_result,
            content_hash=content_hash(
                thesis_data.title,
                thesis_data.description,
                thesis_data.scope,
                thesis_data.objectives,
                thesis_data.expected_result,
                thesis_data.semester,
                thesis_data.program,
            ),
            needs_review=bool(candidates),
        )
        self.repo.create(thesis)
        self._attach_classifications(thesis, thesis_data)
        self.db.flush()
        detail = "Imported thesis"
        if candidates:
            detail += "; marked for review due to similar content"
        self.audit.log("SAVE_THESIS", "thesis", str(thesis.thesis_id), detail)
        return thesis.thesis_id

    def import_rows(self, rows: list[dict], semester: str | None = None, program: str | None = None):
        new_ids: list[uuid.UUID] = []
        errors: list[dict] = []
        self.audit.log("IMPORT_EXCEL", "thesis", None, f"Received {len(rows)} parsed rows")

        for raw in rows:
            try:
                with self.db.begin_nested():
                    thesis_id = self._persist_row(raw, semester, program)
                    if thesis_id is not None:
                        new_ids.append(thesis_id)
            except (ValueError, SQLAlchemyError) as exc:
                self.db.rollback()
                self.audit.log("IMPORT_ERROR", "thesis", None, f"Row {raw.get('row_number')}: {exc}")
                errors.append({"row": raw.get("row_number"), "error": str(exc)})

        if new_ids:
            self.similarity_service.run_for_new(new_ids)
            self.audit.log("RUN_SIMILARITY", "similarity", None, f"Calculated similarity for {len(new_ids)} new theses")
        self.db.commit()
        return new_ids, errors

    def create_thesis(
        self, payload: dict, thesis_id: uuid.UUID, semester: str | None = None, program: str | None = None
    ) -> uuid.UUID | None:
        """Add a single topic submitted through the API and score it against the corpus.

        ``thesis_id`` is the web system's project id, reused verbatim as the thesis id.
        """
        raw = {"sheet_name": "api", "row_number": 1, "data": payload}
        self.audit.log("CREATE_THESIS", "thesis", str(thesis_id), "Received one topic via API")
        try:
            thesis_id = self._persist_row(raw, semester, program, thesis_id)
        except (ValueError, SQLAlchemyError):
            self.db.rollback()
            raise
        if thesis_id is not None:
            self.similarity_service.run_for_new([thesis_id])
            self.audit.log("RUN_SIMILARITY", "similarity", None, "Calculated similarity for 1 new thesis")
        self.db.commit()
        return thesis_id
