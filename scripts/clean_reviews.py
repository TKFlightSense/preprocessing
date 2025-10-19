#!/usr/bin/env python3
"""
Clean special characters from a CSV with a single `review` column.

Defaults:
- Keeps only ASCII letters (A-Z a-z), digits (0-9), and spaces
- Collapses multiple whitespace to a single space, strips leading/trailing spaces

Options allow keeping common punctuation, or keeping ONLY periods.

Usage:
  python scripts/clean_reviews.py --input CLEAN_BA_AirlineReviews.csv --output CLEAN_BA_AirlineReviews.clean.csv

Optional flags:
  --column review                 # column name (default: review)
  --keep-punct                    # keep . , ! ? ; : ' " - ( ) characters
  --only-dots                     # delete everything except '.' (periods)
  --inplace                       # overwrite input file in place
"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd


def clean_text(text: str, keep_punct: bool = False, only_dots: bool = False) -> str:
    if text is None:
        return ""
    s = str(text)
    # Normalize and strip diacritics to get ASCII when possible
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", "ignore").decode("ascii")

    if only_dots:
        # Keep only periods. Remove everything else.
        s = re.sub(r"[^.]+", "", s)
        return s

    if keep_punct:
        # Keep basic punctuation . , ! ? ; : ' " - ( )
        s = re.sub(r"[^A-Za-z0-9\s\.,!\?;:'\"\-\(\)]", "", s)
    else:
        # Keep only letters, digits, whitespace
        s = re.sub(r"[^A-Za-z0-9\s]", "", s)

    # Collapse whitespace and trim
    s = re.sub(r"\s+", " ", s).strip()
    return s


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Remove special characters from review column in a CSV")
    p.add_argument("--input", required=True, help="Input CSV path")
    p.add_argument("--output", help="Output CSV path (omit with --inplace)")
    p.add_argument("--column", default="review", help="Column to clean (default: review)")
    p.add_argument("--keep-punct", action="store_true", help="Keep basic punctuation . , ! ? ; : ' \" - ( )")
    p.add_argument("--only-dots", action="store_true", help="Delete everything except '.' (period) characters")
    p.add_argument("--inplace", action="store_true", help="Overwrite input file in place")
    args = p.parse_args(argv)

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"[ERR] Input not found: {in_path}", file=sys.stderr)
        return 2

    if args.inplace:
        out_path = in_path
    else:
        if not args.output:
            print("[ERR] Provide --output or use --inplace", file=sys.stderr)
            return 2
        out_path = Path(args.output)

    df = pd.read_csv(in_path)
    if args.column not in df.columns:
        return (
            print(f"[ERR] Column '{args.column}' not found. Available: {list(df.columns)}", file=sys.stderr)
            or 2
        )

    df[args.column] = df[args.column].astype(str).map(
        lambda x: clean_text(x, keep_punct=args.keep_punct, only_dots=args.only_dots)
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote cleaned file: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
