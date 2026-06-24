from typing import List, Optional

from pydantic import BaseModel


class SimilarityOut(BaseModel):
    sim_id: int
    thesis_a_id: int
    thesis_b_id: int
    semantic_score: float
    lexical_score: float
    structure_score: float
    domain_score: float
    overall_score: float
    level: str
    reason: Optional[List[str]] = None

    model_config = {"from_attributes": True}
