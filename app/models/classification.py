from sqlalchemy import Column, ForeignKey, Integer, String, Table, UniqueConstraint
from sqlalchemy.orm import relationship

from app.database import Base

thesis_domain = Table(
    "thesis_domain",
    Base.metadata,
    Column("thesis_id", Integer, ForeignKey("thesis.thesis_id"), primary_key=True),
    Column("domain_id", Integer, ForeignKey("domain.domain_id"), primary_key=True),
)

thesis_semantic = Table(
    "thesis_semantic",
    Base.metadata,
    Column("thesis_id", Integer, ForeignKey("thesis.thesis_id"), primary_key=True),
    Column("semantic_id", Integer, ForeignKey("semantic_category.semantic_id"), primary_key=True),
)

thesis_structure = Table(
    "thesis_structure",
    Base.metadata,
    Column("thesis_id", Integer, ForeignKey("thesis.thesis_id"), primary_key=True),
    Column("structure_id", Integer, ForeignKey("structure_type.structure_id"), primary_key=True),
)

thesis_lexical = Table(
    "thesis_lexical",
    Base.metadata,
    Column("thesis_id", Integer, ForeignKey("thesis.thesis_id"), primary_key=True),
    Column("tag_id", Integer, ForeignKey("lexical_tag.tag_id"), primary_key=True),
)

thesis_tech = Table(
    "thesis_tech",
    Base.metadata,
    Column("thesis_id", Integer, ForeignKey("thesis.thesis_id"), primary_key=True),
    Column("tech_id", Integer, ForeignKey("tech.tech_id"), primary_key=True),
)


class Domain(Base):
    __tablename__ = "domain"
    __table_args__ = (UniqueConstraint("name", name="uq_domain_name"),)

    domain_id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    theses = relationship("Thesis", secondary=thesis_domain, back_populates="domains")


class SemanticCategory(Base):
    __tablename__ = "semantic_category"
    __table_args__ = (UniqueConstraint("name", name="uq_semantic_name"),)

    semantic_id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    theses = relationship("Thesis", secondary=thesis_semantic, back_populates="semantics")


class StructureType(Base):
    __tablename__ = "structure_type"
    __table_args__ = (UniqueConstraint("name", name="uq_structure_name"),)

    structure_id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    theses = relationship("Thesis", secondary=thesis_structure, back_populates="structures")


class LexicalTag(Base):
    __tablename__ = "lexical_tag"
    __table_args__ = (UniqueConstraint("name", name="uq_lexical_name"),)

    tag_id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    theses = relationship("Thesis", secondary=thesis_lexical, back_populates="lexical_tags")


class Tech(Base):
    __tablename__ = "tech"
    __table_args__ = (UniqueConstraint("name", name="uq_tech_name"),)

    tech_id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    theses = relationship("Thesis", secondary=thesis_tech, back_populates="technologies")
