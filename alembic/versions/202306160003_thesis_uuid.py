"""thesis_id integer -> uuid

Revision ID: 202306160003
Revises: 202306160002
Create Date: 2026-08-05 00:00:00

Đổi khoá chính thesis.thesis_id (và mọi khoá ngoại tham chiếu tới nó) từ integer
sang uuid, để một đề tài dùng chung id với project bên hệ thống web. Vì id integer
cũ không map được sang uuid nên dữ liệu thesis/similarity cũ (dữ liệu test) bị xoá.
audit_log.record_id đổi sang varchar để chứa được uuid.
"""

from alembic import op

revision = "202306160003"
down_revision = "202306160002"
branch_labels = None
depends_on = None

# (table, foreign-key constraint name, column) — Postgres default names from the
# inline ForeignKey definitions in the initial migration.
_THESIS_FKS = [
    ("similarity", "similarity_thesis_a_id_fkey", "thesis_a_id"),
    ("similarity", "similarity_thesis_b_id_fkey", "thesis_b_id"),
    ("thesis_domain", "thesis_domain_thesis_id_fkey", "thesis_id"),
    ("thesis_semantic", "thesis_semantic_thesis_id_fkey", "thesis_id"),
    ("thesis_structure", "thesis_structure_thesis_id_fkey", "thesis_id"),
    ("thesis_lexical", "thesis_lexical_thesis_id_fkey", "thesis_id"),
    ("thesis_tech", "thesis_tech_thesis_id_fkey", "thesis_id"),
]


def upgrade():
    # 1. Drop every FK that references thesis.thesis_id (can't retype a referenced column otherwise).
    for table, fk_name, _ in _THESIS_FKS:
        op.execute(f'ALTER TABLE "{table}" DROP CONSTRAINT IF EXISTS "{fk_name}"')

    # 2. Wipe the incompatible integer-keyed rows (old test data; ids are meaningless as uuids).
    op.execute(
        "TRUNCATE TABLE similarity, thesis_domain, thesis_semantic, "
        "thesis_structure, thesis_lexical, thesis_tech, thesis RESTART IDENTITY CASCADE"
    )

    # 3. thesis PK: drop the integer sequence default, retype to uuid (client-supplied).
    op.execute("ALTER TABLE thesis ALTER COLUMN thesis_id DROP DEFAULT")
    op.execute("ALTER TABLE thesis ALTER COLUMN thesis_id TYPE uuid USING (gen_random_uuid())")
    op.execute("DROP SEQUENCE IF EXISTS thesis_thesis_id_seq")

    # 4. Retype every referencing column to uuid.
    for table, _, column in _THESIS_FKS:
        op.execute(f'ALTER TABLE "{table}" ALTER COLUMN "{column}" TYPE uuid USING (gen_random_uuid())')

    # 5. Recreate the FKs.
    for table, fk_name, column in _THESIS_FKS:
        op.execute(
            f'ALTER TABLE "{table}" ADD CONSTRAINT "{fk_name}" '
            f'FOREIGN KEY ("{column}") REFERENCES thesis (thesis_id)'
        )

    # 6. audit_log.record_id: integer -> varchar so it can hold a thesis uuid.
    op.execute("ALTER TABLE audit_log ALTER COLUMN record_id TYPE varchar USING (record_id::varchar)")


def downgrade():
    # Best-effort reverse. Data is not restored (the uuid rows can't map back to the old integers).
    op.execute("ALTER TABLE audit_log ALTER COLUMN record_id TYPE integer USING (NULLIF(record_id, '')::integer)")

    for table, fk_name, _ in _THESIS_FKS:
        op.execute(f'ALTER TABLE "{table}" DROP CONSTRAINT IF EXISTS "{fk_name}"')

    op.execute(
        "TRUNCATE TABLE similarity, thesis_domain, thesis_semantic, "
        "thesis_structure, thesis_lexical, thesis_tech, thesis RESTART IDENTITY CASCADE"
    )

    op.execute("CREATE SEQUENCE IF NOT EXISTS thesis_thesis_id_seq")
    op.execute("ALTER TABLE thesis ALTER COLUMN thesis_id TYPE integer USING (0)")
    op.execute("ALTER TABLE thesis ALTER COLUMN thesis_id SET DEFAULT nextval('thesis_thesis_id_seq')")
    op.execute("ALTER SEQUENCE thesis_thesis_id_seq OWNED BY thesis.thesis_id")

    for table, _, column in _THESIS_FKS:
        op.execute(f'ALTER TABLE "{table}" ALTER COLUMN "{column}" TYPE integer USING (0)')

    for table, fk_name, column in _THESIS_FKS:
        op.execute(
            f'ALTER TABLE "{table}" ADD CONSTRAINT "{fk_name}" '
            f'FOREIGN KEY ("{column}") REFERENCES thesis (thesis_id)'
        )
