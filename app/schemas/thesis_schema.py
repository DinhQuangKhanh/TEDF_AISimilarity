from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class ThesisBase(BaseModel):
    semester: Optional[str] = None
    program: Optional[str] = None
    title: Optional[str] = None
    description: Optional[str] = None
    scope: Optional[str] = None
    objectives: Optional[str] = None
    expected_result: Optional[str] = None


class ThesisCreateRequest(BaseModel):
    """One topic submitted through the API (all five content fields).

    ``thesis_id`` is supplied by the caller (the web system's project id) so the
    thesis and its project share one identifier.
    """

    thesis_id: UUID
    title: str = Field(min_length=1)
    description: Optional[str] = None
    scope: Optional[str] = None
    objectives: Optional[str] = None
    expected_result: Optional[str] = None
    semester: Optional[str] = None
    program: Optional[str] = None
    domains: List[str] = Field(default_factory=list)
    technologies: List[str] = Field(default_factory=list)

    def to_raw_data(self) -> dict:
        """Shape it like a parsed Excel row so the import pipeline can be reused."""
        return {
            "title": self.title,
            "description": self.description,
            "scope": self.scope,
            "objectives": self.objectives,
            "expected_result": self.expected_result,
            "semester": self.semester,
            "program": self.program,
            "domains": ", ".join(self.domains) or None,
            "technologies": ", ".join(self.technologies) or None,
        }


class ThesisCreate(ThesisBase):
    domains: List[str] = Field(default_factory=list)
    semantic_categories: List[str] = Field(default_factory=list)
    structure_types: List[str] = Field(default_factory=list)
    lexical_tags: List[str] = Field(default_factory=list)
    technologies: List[str] = Field(default_factory=list)


class ThesisListItem(ThesisBase):
    thesis_id: UUID

    model_config = {"from_attributes": True}


class ThesisDetail(ThesisBase):
    thesis_id: UUID
    domains: List[str] = Field(default_factory=list)
    semantic_categories: List[str] = Field(default_factory=list)
    structure_types: List[str] = Field(default_factory=list)
    lexical_tags: List[str] = Field(default_factory=list)
    technologies: List[str] = Field(default_factory=list)

    model_config = {"from_attributes": True}
