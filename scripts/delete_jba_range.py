#!/usr/bin/env python3
"""Remove JBA rows for a specific date range from JunDash Supabase."""

from __future__ import annotations

import argparse
import os
from datetime import date

from sqlalchemy import create_engine, text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Delete JBA rows between two dates")
    parser.add_argument("start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("end", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--db-url",
        default=os.environ.get("SUPABASE_DB_URL"),
        help="Supabase connection string",
    )
    parser.add_argument("--studio", default="JBA", help="Studio code (default: JBA)")
    parser.add_argument("--dry-run", action="store_true", help="Preview count only")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not args.db_url:
        parser.error("Set SUPABASE_DB_URL or pass --db-url")

    engine = create_engine(args.db_url, connect_args={"sslmode": "require"})
    select_stmt = text(
        """
        SELECT count(*) FROM public.studio_daily_metrics
        WHERE studio = :studio AND date BETWEEN :start AND :end
        """
    )
    delete_stmt = text(
        """
        DELETE FROM public.studio_daily_metrics
        WHERE studio = :studio AND date BETWEEN :start AND :end
        """
    )
    params = {"studio": args.studio, "start": args.start, "end": args.end}
    with engine.begin() as conn:
        count = conn.execute(select_stmt, params).scalar() or 0
        print(f"Found {count} rows for {args.studio} between {args.start} and {args.end}")
        if not args.dry_run and count:
            deleted = conn.execute(delete_stmt, params).rowcount
            print(f"Deleted {deleted} rows")


if __name__ == "__main__":
    main()
