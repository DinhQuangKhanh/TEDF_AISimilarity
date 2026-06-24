"""create full schema

Revision ID: 202306160001
Revises:
Create Date: 2026-06-16 00:00:00
"""

from alembic import op
import sqlalchemy as sa

revision = "202306160001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "thesis",
        sa.Column("thesis_id", sa.Integer, primary_key=True),
        sa.Column("semester", sa.String, nullable=True),
        sa.Column("program", sa.String, nullable=True),
        sa.Column("title_en", sa.String, nullable=True),
        sa.Column("title_vn", sa.String, nullable=True),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("scope", sa.Text, nullable=True),
        sa.Column("notes", sa.Text, nullable=True),
        sa.Column("is_deleted", sa.Boolean, nullable=False, server_default=sa.false()),
        sa.Column("deleted_at", sa.DateTime, nullable=True),
        sa.Column("needs_review", sa.Boolean, nullable=False, server_default=sa.false()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("title_en", "title_vn", "semester", "program", name="uq_thesis_full"),
    )
    op.create_index("ix_thesis_thesis_id", "thesis", ["thesis_id"])

    op.create_table(
        "domain",
        sa.Column("domain_id", sa.Integer, primary_key=True),
        sa.Column("name", sa.String, nullable=False),
        sa.UniqueConstraint("name", name="uq_domain_name"),
    )
    op.create_index("ix_domain_domain_id", "domain", ["domain_id"])

    op.create_table(
        "semantic_category",
        sa.Column("semantic_id", sa.Integer, primary_key=True),
        sa.Column("name", sa.String, nullable=False),
        sa.UniqueConstraint("name", name="uq_semantic_name"),
    )
    op.create_index("ix_semantic_category_semantic_id", "semantic_category", ["semantic_id"])

    op.create_table(
        "structure_type",
        sa.Column("structure_id", sa.Integer, primary_key=True),
        sa.Column("name", sa.String, nullable=False),
        sa.UniqueConstraint("name", name="uq_structure_name"),
    )
    op.create_index("ix_structure_type_structure_id", "structure_type", ["structure_id"])

    op.create_table(
        "lexical_tag",
        sa.Column("tag_id", sa.Integer, primary_key=True),
        sa.Column("name", sa.String, nullable=False),
        sa.UniqueConstraint("name", name="uq_lexical_name"),
    )
    op.create_index("ix_lexical_tag_tag_id", "lexical_tag", ["tag_id"])

    op.create_table(
        "tech",
        sa.Column("tech_id", sa.Integer, primary_key=True),
        sa.Column("name", sa.String, nullable=False),
        sa.UniqueConstraint("name", name="uq_tech_name"),
    )
    op.create_index("ix_tech_tech_id", "tech", ["tech_id"])

    op.create_table(
        "thesis_domain",
        sa.Column("thesis_id", sa.Integer, sa.ForeignKey("thesis.thesis_id"), primary_key=True),
        sa.Column("domain_id", sa.Integer, sa.ForeignKey("domain.domain_id"), primary_key=True),
    )
    op.create_table(
        "thesis_semantic",
        sa.Column("thesis_id", sa.Integer, sa.ForeignKey("thesis.thesis_id"), primary_key=True),
        sa.Column("semantic_id", sa.Integer, sa.ForeignKey("semantic_category.semantic_id"), primary_key=True),
    )
    op.create_table(
        "thesis_structure",
        sa.Column("thesis_id", sa.Integer, sa.ForeignKey("thesis.thesis_id"), primary_key=True),
        sa.Column("structure_id", sa.Integer, sa.ForeignKey("structure_type.structure_id"), primary_key=True),
    )
    op.create_table(
        "thesis_lexical",
        sa.Column("thesis_id", sa.Integer, sa.ForeignKey("thesis.thesis_id"), primary_key=True),
        sa.Column("tag_id", sa.Integer, sa.ForeignKey("lexical_tag.tag_id"), primary_key=True),
    )
    op.create_table(
        "thesis_tech",
        sa.Column("thesis_id", sa.Integer, sa.ForeignKey("thesis.thesis_id"), primary_key=True),
        sa.Column("tech_id", sa.Integer, sa.ForeignKey("tech.tech_id"), primary_key=True),
    )

    op.create_table(
        "similarity",
        sa.Column("sim_id", sa.Integer, primary_key=True),
        sa.Column("thesis_a_id", sa.Integer, sa.ForeignKey("thesis.thesis_id"), nullable=False),
        sa.Column("thesis_b_id", sa.Integer, sa.ForeignKey("thesis.thesis_id"), nullable=False),
        sa.Column("semantic_score", sa.Float, nullable=False),
        sa.Column("lexical_score", sa.Float, nullable=False),
        sa.Column("structure_score", sa.Float, nullable=False),
        sa.Column("domain_score", sa.Float, nullable=False),
        sa.Column("overall_score", sa.Float, nullable=False),
        sa.Column("level", sa.String, nullable=False),
        sa.Column("reason", sa.String, nullable=True),
        sa.UniqueConstraint("thesis_a_id", "thesis_b_id", name="uq_similarity_pair"),
    )
    op.create_index("ix_similarity_sim_id", "similarity", ["sim_id"])

    op.create_table(
        "audit_log",
        sa.Column("log_id", sa.Integer, primary_key=True),
        sa.Column("timestamp", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("action", sa.String, nullable=False),
        sa.Column("table_name", sa.String, nullable=False),
        sa.Column("record_id", sa.Integer, nullable=True),
        sa.Column("detail", sa.Text, nullable=True),
    )
    op.create_index("ix_audit_log_log_id", "audit_log", ["log_id"])


def downgrade():
    op.drop_index("ix_audit_log_log_id", table_name="audit_log")
    op.drop_table("audit_log")
    op.drop_index("ix_similarity_sim_id", table_name="similarity")
    op.drop_table("similarity")
    op.drop_table("thesis_tech")
    op.drop_table("thesis_lexical")
    op.drop_table("thesis_structure")
    op.drop_table("thesis_semantic")
    op.drop_table("thesis_domain")
    op.drop_index("ix_tech_tech_id", table_name="tech")
    op.drop_table("tech")
    op.drop_index("ix_lexical_tag_tag_id", table_name="lexical_tag")
    op.drop_table("lexical_tag")
    op.drop_index("ix_structure_type_structure_id", table_name="structure_type")
    op.drop_table("structure_type")
    op.drop_index("ix_semantic_category_semantic_id", table_name="semantic_category")
    op.drop_table("semantic_category")
    op.drop_index("ix_domain_domain_id", table_name="domain")
    op.drop_table("domain")
    op.drop_index("ix_thesis_thesis_id", table_name="thesis")
    op.drop_table("thesis")
