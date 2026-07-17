from app.schemas.import_schema import NormalizedRow
from app.schemas.thesis_schema import ThesisCreate
from app.utils.text_cleaner import clean_text, normalize_list


class SchemaMapperService:
    def map(self, normalized: NormalizedRow) -> ThesisCreate:
        return ThesisCreate(
            semester=clean_text(normalized.semester),
            program=clean_text(normalized.program),
            title=clean_text(normalized.title),
            description=clean_text(normalized.description),
            scope=clean_text(normalized.scope),
            objectives=clean_text(normalized.objectives),
            expected_result=clean_text(normalized.expected_result),
            domains=normalize_list(normalized.domains),
            semantic_categories=normalize_list(normalized.semantic_categories),
            structure_types=normalize_list(normalized.structure_types),
            lexical_tags=normalize_list(normalized.lexical_tags),
            technologies=normalize_list(normalized.technologies),
        )
