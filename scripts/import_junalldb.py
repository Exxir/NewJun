#!/usr/bin/env python3
"""Import Jungle legacy DB CSV into Supabase."""

from __future__ import annotations

import argparse
import logging
import os
from typing import Dict, Iterable, List, cast

import pandas as pd
from pandas._typing import IndexLabel
from sqlalchemy import Column, Date, Integer, MetaData, Numeric, String, Table

from scripts.import_jfw_metrics import normalize_columns, parse_numeric, upsert_rows


DEFAULT_TABLE_NAME = "junalldb_metrics"
DATE_COLUMNS = ["date"]
INT_COLUMNS = [
    "class_mat",
    "class_ref",
    "mt_visits_ref",
    "total_visits_mat",
    "cp_visits_mat",
    "cp_visits_ref",
    "first_time_mat",
    "first_time_ref",
]
NUMERIC_COLUMNS = [
    "cp_sales_mat",
    "cp_sales_ref",
    "mt_sales_mat",
    "mt_sales_ref",
    "mt_sales_total",
]
TEXT_COLUMNS = ["studio", "id"]

COLUMN_MAP: Dict[str, str] = {
    "date": "date",
    "studio": "studio",
    "class_mat": "class_mat",
    "class_ref": "class_ref",
    "mt_visits_ref": "mt_visits_ref",
    "total_visits_mat": "total_visits_mat",
    "cp_visits_mat": "cp_visits_mat",
    "cp_visit_ref": "cp_visits_ref",
    "ft_mat": "first_time_mat",
    "ft_ref": "first_time_ref",
    "cp_sales_mat": "cp_sales_mat",
    "cp_sales_ref": "cp_sales_ref",
    "mt_sales_mat": "mt_sales_mat",
    "mt_sales_ref": "mt_sales_ref",
    "mt_sales_total": "mt_sales_total",
    "id": "id",
}


def clean_junalldb_frame(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    df.columns = normalize_columns(df.columns)
    df = df.rename(columns=COLUMN_MAP)

    missing = set(COLUMN_MAP.values()) - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {', '.join(sorted(missing))}")

    for col in DATE_COLUMNS:
        df[col] = pd.to_datetime(df[col], errors="coerce", format="%m/%d/%Y").dt.date

    for col in TEXT_COLUMNS:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({"": None})

    numeric_columns: Iterable[str] = INT_COLUMNS + NUMERIC_COLUMNS
    for col in numeric_columns:
        df[col] = df[col].apply(parse_numeric)

    for col in INT_COLUMNS:
        df[col] = df[col].apply(lambda x: int(x) if pd.notna(x) else None)

    df = df[list(COLUMN_MAP.values())]
    subset_required: IndexLabel = cast(IndexLabel, ["date", "studio", "id"])
    df = df.dropna(subset=subset_required)  # type: ignore[arg-type]
    df = df.astype(object).where(pd.notna(df), None)
    return df


def build_junalldb_table(metadata: MetaData, table_name: str) -> Table:
    return Table(
        table_name,
        metadata,
        Column("id", String, primary_key=True),
        Column("date", Date, nullable=False),
        Column("studio", String, nullable=False),
        Column("class_mat", Integer),
        Column("class_ref", Integer),
        Column("mt_visits_ref", Integer),
        Column("total_visits_mat", Integer),
        Column("cp_visits_mat", Integer),
        Column("cp_visits_ref", Integer),
        Column("first_time_mat", Integer),
        Column("first_time_ref", Integer),
        Column("cp_sales_mat", Numeric(18, 2)),
        Column("cp_sales_ref", Numeric(18, 2)),
        Column("mt_sales_mat", Numeric(18, 2)),
        Column("mt_sales_ref", Numeric(18, 2)),
        Column("mt_sales_total", Numeric(18, 2)),
        schema="public",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Import junalldb.csv into Supabase")
    parser.add_argument(
        "--csv",
        default=os.path.join(
            "/Users/mnaz/Library/CloudStorage/OneDrive-SharedLibraries-ExxirCapital",
            "Exxir Concepts - Documents/Jungle/Dashboards/Databases/junalldb.csv",
        ),
        help="Path to junalldb CSV export",
    )
    parser.add_argument(
        "--db-url",
        default=os.environ.get("SUPABASE_DB_URL"),
        help="Supabase/Postgres connection string",
    )
    parser.add_argument(
        "--table-name",
        default=DEFAULT_TABLE_NAME,
        help="Destination table name (default: junalldb_metrics)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Parse but do not write to DB")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")

    if not os.path.exists(args.csv):
        parser.error(f"CSV path not found: {args.csv}")
    if not args.db_url and not args.dry_run:
        parser.error("SUPABASE_DB_URL must be provided unless using --dry-run")

    raw_df = pd.read_csv(args.csv)
    cleaned = clean_junalldb_frame(raw_df)
    logging.info("Parsed %d rows from %s", len(cleaned), args.csv)

    if args.dry_run:
        logging.info("Dry run complete. Sample rows:\n%s", cleaned.head())
        return

    metadata = MetaData()
    table = build_junalldb_table(metadata, args.table_name)
    records: List[Dict[str, object]] = cleaned.to_dict("records")
    inserted = upsert_rows(table, records, args.db_url) if records else 0
    logging.info("Upserted %d rows into %s", inserted, args.table_name)


if __name__ == "__main__":
    main()
