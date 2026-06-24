from sqlalchemy import Boolean, Column, DateTime, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.orm import relationship

from app.database import Base
from app.models.classification import thesis_domain, thesis_lexical, thesis_semantic, thesis_structure, thesis_tech


class Thesis(Base):
    __tablename__ = "thesis"
    __table_args__ = (
        UniqueConstraint("title_en", "title_vn", "semester", "program", name="uq_thesis_full"),
    )

    thesis_id = Column(Integer, primary_key=True, index=True)
    semester = Column(String, nullable=True)
    program = Column(String, nullable=True)
    title_en = Column(String, nullable=True)
    title_vn = Column(String, nullable=True)
    description = Column(Text, nullable=True)
    scope = Column(Text, nullable=True)
    notes = Column(Text, nullable=True)
    is_deleted = Column(Boolean, default=False, nullable=False)
    deleted_at = Column(DateTime, nullable=True)
    needs_review = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    domains = relationship("Domain", secondary=thesis_domain, back_populates="theses")
    semantics = relationship("SemanticCategory", secondary=thesis_semantic, back_populates="theses")
    structures = relationship("StructureType", secondary=thesis_structure, back_populates="theses")
    lexical_tags = relationship("LexicalTag", secondary=thesis_lexical, back_populates="theses")
    technologies = relationship("Tech", secondary=thesis_tech, back_populates="theses")
    similarities_a = relationship("Similarity", foreign_keys="Similarity.thesis_a_id", back_populates="thesis_a")
    similarities_b = relationship("Similarity", foreign_keys="Similarity.thesis_b_id", back_populates="thesis_b")
