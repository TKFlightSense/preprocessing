#!/usr/bin/env python3
"""
Clean special characters from a CSV with a single `review` column.

Defaults:
- Keeps only ASCII letters (A-Z a-z), digits (0-9), and spaces
- Collapses multiple whitespace to a single space, strips leading/trailing spaces

Options allow keeping common punctuation.
Optional flags:
  --column review                 # column name (default: review)
  --keep-punct                    # keep . , ! ? ; : ' " - ( ) characters
  --inplace                       # overwrite input file in place
"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd


def clean_text(text: str, keep_punct: bool = False) -> str:
    if text is None:
        return ""
    s = str(text)
    # Normalize and strip diacritics to get ASCII when possible
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", "ignore").decode("ascii")

    s = re.sub(r"[^A-Za-z0-9\s\.]", "", s)

    s = re.sub(r"\s+", " ", s).strip()
    return s


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Clean the review column in a CSV by removing all punctuation except periods ('.')."
    )
    parser.add_argument("--input", required=True, help="Path to input CSV file.")
    parser.add_argument(
        "--output", help="Path to output CSV file (omit with --inplace)."
    )
    parser.add_argument(
        "--column",
        default="review",
        help="Name of the column to clean (default: 'review').",
    )
    parser.add_argument(
        "--inplace", action="store_true", help="Overwrite the input file directly."
    )

    args = parser.parse_args(argv)

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"[ERR] Input not found: {in_path}", file=sys.stderr)
        return 2

    if args.inplace:
        out_path = in_path
    elif args.output:
        out_path = Path(args.output)
    else:
        print("[ERR] Provide --output or use --inplace", file=sys.stderr)
        return 2

    df = pd.read_csv(in_path)
    if args.column not in df.columns:
        print(
            f"[ERR] Column '{args.column}' not found. Available: {list(df.columns)}",
            file=sys.stderr,
        )
        return 2

    df[args.column] = df[args.column].astype(str).map(clean_text)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[OK] Wrote cleaned file: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
