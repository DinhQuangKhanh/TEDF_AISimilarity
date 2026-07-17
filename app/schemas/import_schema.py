from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class ApiResponse(BaseModel):
    success: bool
    message: str
    data: dict = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    success: bool = False
    message: str
    errors: list = Field(default_factory=list)


class NormalizedRow(BaseModel):
    semester: Optional[str] = None
    program: Optional[str] = None
    title: Optional[str] = None
    description: Optional[str] = None
    scope: Optional[str] = None
    objectives: Optional[str] = None
    expected_result: Optional[str] = None
    domains: List[str] = Field(default_factory=list)
    semantic_categories: List[str] = Field(default_factory=list)
    structure_types: List[str] = Field(default_factory=list)
    lexical_tags: List[str] = Field(default_factory=list)
    technologies: List[str] = Field(default_factory=list)

    @field_validator(
        "domains",
        "semantic_categories",
        "structure_types",
        "lexical_tags",
        "technologies",
        mode="before",
    )
    @classmethod
    def ensure_list(cls, value):
        return value or []
