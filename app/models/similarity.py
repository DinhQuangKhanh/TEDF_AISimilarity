from sqlalchemy import Column, Float, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import relationship

from app.database import Base


class Similarity(Base):
    __tablename__ = "similarity"
    __table_args__ = (UniqueConstraint("thesis_a_id", "thesis_b_id", name="uq_similarity_pair"),)

    sim_id = Column(Integer, primary_key=True, index=True)
    thesis_a_id = Column(Integer, ForeignKey("thesis.thesis_id"), nullable=False)
    thesis_b_id = Column(Integer, ForeignKey("thesis.thesis_id"), nullable=False)
    semantic_score = Column(Float, nullable=False)
    lexical_score = Column(Float, nullable=False)
    structure_score = Column(Float, nullable=False)
    domain_score = Column(Float, nullable=False)
    overall_score = Column(Float, nullable=False)
    level = Column(String, nullable=False)
    reason = Column(String, nullable=True)

    thesis_a = relationship("Thesis", foreign_keys=[thesis_a_id], back_populates="similarities_a")
    thesis_b = relationship("Thesis", foreign_keys=[thesis_b_id], back_populates="similarities_b")
