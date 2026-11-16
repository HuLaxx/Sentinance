"""Add replay-safe raw Kafka price event ledger.

Revision ID: 20261116_price_v1
Revises: 11f95b7cd4f4
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20261116_price_v1"
down_revision: str | None = "11f95b7cd4f4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "raw_kafka_price_events",
        sa.Column("topic", sa.String(length=200), nullable=False),
        sa.Column("partition", sa.Integer(), nullable=False),
        sa.Column("offset", sa.BigInteger(), nullable=False),
        sa.Column("schema_version", sa.Integer(), nullable=False),
        sa.Column("source", sa.String(length=32), nullable=False),
        sa.Column("source_event_id", sa.String(length=128), nullable=False),
        sa.Column("symbol", sa.String(length=20), nullable=False),
        sa.Column("observed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("price", sa.Numeric(precision=30, scale=12), nullable=False),
        sa.Column("volume", sa.Numeric(precision=30, scale=12), nullable=False),
        sa.Column("raw_event", sa.JSON(), nullable=False),
        sa.Column("received_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("topic", "partition", "offset"),
        sa.UniqueConstraint(
            "source", "source_event_id", name="uq_kafka_price_source_event"
        ),
    )
    op.create_index(
        "ix_raw_kafka_price_symbol_time",
        "raw_kafka_price_events",
        ["symbol", "observed_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_raw_kafka_price_symbol_time", table_name="raw_kafka_price_events")
    op.drop_table("raw_kafka_price_events")
