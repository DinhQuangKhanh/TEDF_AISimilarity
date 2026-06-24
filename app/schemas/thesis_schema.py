from typing import List, Optional

from pydantic import BaseModel, Field


class ThesisBase(BaseModel):
    semester: Optional[str] = None
    program: Optional[str] = None
    title_en: Optional[str] = None
    title_vn: Optional[str] = None
    description: Optional[str] = None
    scope: Optional[str] = None
    notes: Optional[str] = None


class ThesisCreate(ThesisBase):
    domains: List[str] = Field(default_factory=list)
    semantic_categories: List[str] = Field(default_factory=list)
    structure_types: List[str] = Field(default_factory=list)
    lexical_tags: List[str] = Field(default_factory=list)
    technologies: List[str] = Field(default_factory=list)


class ThesisListItem(ThesisBase):
    thesis_id: int

    model_config = {"from_attributes": True}


class ThesisDetail(ThesisBase):
    thesis_id: int
    domains: List[str] = Field(default_factory=list)
    semantic_categories: List[str] = Field(default_factory=list)
    structure_types: List[str] = Field(default_factory=list)
    lexical_tags: List[str] = Field(default_factory=list)
    technologies: List[str] = Field(default_factory=list)

    model_config = {"from_attributes": True}
