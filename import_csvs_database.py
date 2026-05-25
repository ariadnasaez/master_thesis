"""
Imports all CSV files from this folder into the MySQL database `tfm_datanex`.
Each CSV becomes a table named after its filename (without .csv).
Column types are inferred by pandas; columns are created as nullable.
"""

import os
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, types as satypes

CSV_DIR = Path(__file__).parent
DB_NAME = "tfm_datanex"

CONN_STR = f"mysql+pymysql://root@127.0.0.1:3306/{DB_NAME}?charset=utf8mb4"
engine = create_engine(CONN_STR)


def dtype_map(df: pd.DataFrame) -> dict:
    """Map pandas dtypes -> SQLAlchemy types appropriate for MySQL."""
    mapping = {}
    for col, dtype in df.dtypes.items():
        kind = dtype.kind
        if kind in ("i", "u"):
            mapping[col] = satypes.BigInteger()
        elif kind == "f":
            mapping[col] = satypes.Float()
        elif kind == "M":
            mapping[col] = satypes.DateTime()
        elif kind == "b":
            mapping[col] = satypes.Boolean()
        else:
            # Strings: use TEXT to be safe for long free-text fields.
            max_len = 0
            try:
                non_null = df[col].dropna().astype(str)
                if len(non_null):
                    max_len = int(non_null.map(len).max())
            except Exception:
                max_len = 0
            if max_len <= 255 and max_len > 0:
                mapping[col] = satypes.String(255)
            else:
                mapping[col] = satypes.Text()
    return mapping


def load_csv(path: Path) -> pd.DataFrame:
    """Read a CSV with BOM-tolerant encoding and try to parse common date columns."""
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    # Strip BOM/whitespace from column names just in case
    df.columns = [c.strip().lstrip("﻿") for c in df.columns]

    # Auto-parse columns whose name suggests a date/timestamp.
    date_hints = ("date", "timestamp")
    for col in df.columns:
        lc = col.lower()
        if any(h in lc for h in date_hints):
            try:
                df[col] = pd.to_datetime(df[col], errors="coerce")
            except Exception:
                pass
    return df


def main() -> int:
    csv_files = sorted(CSV_DIR.glob("*.csv"))
    if not csv_files:
        print("No CSV files found.")
        return 1

    summary = []
    for csv_path in csv_files:
        table = csv_path.stem
        print(f"-> Loading {csv_path.name} ...", flush=True)
        df = load_csv(csv_path)
        rows, cols = df.shape
        print(f"   {rows:,} rows x {cols} cols", flush=True)

        df.to_sql(
            name=table,
            con=engine,
            if_exists="replace",
            index=False,
            chunksize=2000,
            method="multi",
            dtype=dtype_map(df),
        )
        summary.append((table, rows, cols))
        print(f"   OK -> table `{table}`", flush=True)

    print("\n=== Import summary ===")
    for table, rows, cols in summary:
        print(f"  {table:<35} {rows:>10,} rows  {cols:>3} cols")
    print(f"Total tables: {len(summary)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
