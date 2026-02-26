#!/usr/bin/env python3
"""Scrape Jungle Studio Fitness site metrics and load them into Supabase."""

from __future__ import annotations

import argparse
import logging
import os
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup
from sqlalchemy import MetaData

from scripts.import_jfw_metrics import (
    DEFAULT_TABLE_NAME,
    build_table,
    clean_frame,
    normalize_columns,
    upsert_rows,
)


DEFAULT_BASE_URL = "https://junglestudiofitness.marianatek.com/"
DEFAULT_LOGIN_PATH = "/users/sign_in"
DEFAULT_METRICS_PATH = "/reports/studio_daily"
DEFAULT_TABLE_SELECTOR = "table#studio-daily-metrics"

# Maps normalized column labels from the HTML table to the importer column names.
COLUMN_RENAMES: Mapping[str, str] = {
    "mtvisits": "mt_visits",
    "cpvisits": "cp_visits",
    "totalvisits": "total_visits",
    "estvisits": "est_visits",
    "firsttime": "first_time",
    "netsales": "net_sales",
    "mtsales": "mt_sales",
    "cpsales": "cp_sales",
    "occ_pct": "occ_pct",
}


def parse_kv_pairs(pairs: Iterable[str]) -> Dict[str, str]:
    parsed: Dict[str, str] = {}
    for item in pairs:
        if "=" not in item:
            raise ValueError(f"Invalid key=value pair: '{item}'")
        key, value = item.split("=", 1)
        parsed[key] = value
    return parsed


def login(
    session: requests.Session,
    base_url: str,
    login_path: str,
    email: str,
    password: str,
    email_field: str,
    password_field: str,
    extra_fields: Optional[Mapping[str, str]] = None,
) -> None:
    login_url = urljoin(base_url, login_path)
    payload: MutableMapping[str, str] = {
        email_field: email,
        password_field: password,
    }
    if extra_fields:
        payload.update(extra_fields)
    response = session.post(login_url, data=payload, timeout=30)
    response.raise_for_status()
    if "invalid" in response.text.lower():
        raise RuntimeError("Login failed; verify credentials and extra form fields.")


def fetch_metrics(session: requests.Session, base_url: str, metrics_path: str) -> str:
    metrics_url = urljoin(base_url, metrics_path)
    response = session.get(metrics_url, timeout=30)
    response.raise_for_status()
    return response.text


def parse_metrics_table(html: str, selector: str) -> pd.DataFrame:
    soup = BeautifulSoup(html, "html.parser")
    table = soup.select_one(selector)
    if table is None:
        raise RuntimeError(
            f"Could not find metrics table using selector '{selector}'. Update the selector."
        )

    header_cells = table.select("thead tr th") or table.select("tr th")
    headers = [cell.get_text(strip=True) for cell in header_cells]
    normalized_headers = normalize_columns(headers)
    renamed_headers = [COLUMN_RENAMES.get(name, name) for name in normalized_headers]

    rows = table.select("tbody tr") or table.select("tr")[1:]
    records: List[Dict[str, Any]] = []
    for row in rows:
        cells = row.find_all(["td", "th"])
        values = [cell.get_text(strip=True) for cell in cells]
        if not values or all(value == "" for value in values):
            continue
        if len(values) != len(renamed_headers):
            logging.debug("Skipping row with mismatched column count: %s", values)
            continue
        records.append(dict(zip(renamed_headers, values)))

    if not records:
        raise RuntimeError("No rows parsed from metrics table; verify selectors and markup.")

    return pd.DataFrame(records)


def write_output(df: pd.DataFrame, output_path: Optional[str]) -> None:
    if output_path:
        df.to_csv(output_path, index=False)
        logging.info("Wrote raw scrape output to %s", output_path)


def load_dataframe(df: pd.DataFrame, db_url: str, table_name: str, dry_run: bool) -> int:
    cleaned = clean_frame(df)
    if dry_run:
        logging.info("Dry run: %d cleaned rows ready for load", len(cleaned))
        return 0
    records = cleaned.to_dict("records")
    if not records:
        logging.info("No cleaned rows to load.")
        return 0
    metadata = MetaData()
    table = build_table(metadata, table_name)
    inserted = upsert_rows(table, records, db_url)
    logging.info("Upserted %d rows into %s", inserted, table_name)
    return inserted


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scrape Jungle Studio Fitness metrics")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Site base URL")
    parser.add_argument("--login-path", default=DEFAULT_LOGIN_PATH, help="Login form action path")
    parser.add_argument("--metrics-path", default=DEFAULT_METRICS_PATH, help="Path to metrics page")
    parser.add_argument(
        "--table-selector",
        default=DEFAULT_TABLE_SELECTOR,
        help="CSS selector that targets the metrics table",
    )
    parser.add_argument("--email", default=os.environ.get("JUNGLE_DASHBOARD_EMAIL"), help="Login email")
    parser.add_argument(
        "--password",
        default=os.environ.get("JUNGLE_DASHBOARD_PASSWORD"),
        help="Login password",
    )
    parser.add_argument(
        "--email-field",
        default="user[email]",
        help="Form field name for the email/username",
    )
    parser.add_argument(
        "--password-field",
        default="user[password]",
        help="Form field name for the password",
    )
    parser.add_argument(
        "--extra-login-field",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Additional key=value pairs to include in the login POST",
    )
    parser.add_argument(
        "--output",
        help="Write the raw scraped table to this CSV file for inspection",
    )
    parser.add_argument(
        "--db-url",
        default=os.environ.get("SUPABASE_DB_URL"),
        help="Database connection URL",
    )
    parser.add_argument(
        "--table-name",
        default=DEFAULT_TABLE_NAME,
        help="Destination table name in Supabase",
    )
    parser.add_argument("--dry-run", action="store_true", help="Skip loading into the database")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")

    if not args.email or not args.password:
        parser.error("Email and password are required (provide via flags or env vars).")
    if not args.db_url and not args.dry_run:
        parser.error("Database URL required unless --dry-run is set.")

    session = requests.Session()

    extra_fields = parse_kv_pairs(args.extra_login_field)
    login(
        session,
        args.base_url,
        args.login_path,
        args.email,
        args.password,
        args.email_field,
        args.password_field,
        extra_fields,
    )
    logging.info("Authenticated to %s", args.base_url)

    html = fetch_metrics(session, args.base_url, args.metrics_path)
    table_df = parse_metrics_table(html, args.table_selector)
    logging.info("Parsed %d rows from the metrics table", len(table_df))

    write_output(table_df, args.output)

    if args.dry_run:
        load_dataframe(table_df, db_url="", table_name=args.table_name, dry_run=True)
    else:
        load_dataframe(table_df, db_url=args.db_url, table_name=args.table_name, dry_run=False)


if __name__ == "__main__":
    main()
