"""thesis five content fields + content_hash

Revision ID: 202306160002
Revises: 202306160001
Create Date: 2026-07-16 00:00:00

Chuyển bảng thesis về đúng 5 trường nội dung: title, description, scope,
objectives, expected_result. Bỏ title_vn và notes; đổi title_en -> title.
Thêm content_hash (băm 5 trường + semester + program) làm khoá chống trùng.
"""

from alembic import op
import sqlalchemy as sa

revision = "202306160002"
down_revision = "202306160001"
branch_labels = None
depends_on = None


def upgrade():
    # Bỏ ràng buộc unique cũ (dựa trên title_en/title_vn) trước khi drop cột.
    op.drop_constraint("uq_thesis_full", "thesis", type_="unique")

    # title_en -> title (giữ nguyên dữ liệu hiện có).
    op.alter_column("thesis", "title_en", new_column_name="title")

    # Bỏ các cột không còn dùng.
    op.drop_column("thesis", "title_vn")
    op.drop_column("thesis", "notes")

    # Thêm 2 trường nội dung mới + cột băm chống trùng.
    op.add_column("thesis", sa.Column("objectives", sa.Text, nullable=True))
    op.add_column("thesis", sa.Column("expected_result", sa.Text, nullable=True))
    op.add_column("thesis", sa.Column("content_hash", sa.String, nullable=True))

    # Ràng buộc unique mới trên content_hash (NULL vẫn được coi là phân biệt).
    op.create_unique_constraint("uq_thesis_content_hash", "thesis", ["content_hash"])


def downgrade():
    op.drop_constraint("uq_thesis_content_hash", "thesis", type_="unique")
    op.drop_column("thesis", "content_hash")
    op.drop_column("thesis", "expected_result")
    op.drop_column("thesis", "objectives")
    op.add_column("thesis", sa.Column("notes", sa.Text, nullable=True))
    op.add_column("thesis", sa.Column("title_vn", sa.String, nullable=True))
    op.alter_column("thesis", "title", new_column_name="title_en")
    op.create_unique_constraint(
        "uq_thesis_full", "thesis", ["title_en", "title_vn", "semester", "program"]
    )
