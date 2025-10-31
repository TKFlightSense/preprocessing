from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import yaml


def load_labels_yaml(path: str | Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    labels = data.get("labels")
    if not isinstance(labels, list) or not labels:
        raise ValueError("labels.yaml must have a top-level 'labels' list")
    return [str(x).strip() for x in labels]


def read_table(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() in {".csv"}:
        return pd.read_csv(p)
    if p.suffix.lower() in {".jsonl", ".ndjson"}:
        rows = [
            json.loads(line)
            for line in p.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return pd.DataFrame(rows)
    raise ValueError("Unsupported input format. Use .csv or .jsonl")


def write_table(df: pd.DataFrame, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() == ".csv":
        df.to_csv(p, index=False)
    elif p.suffix.lower() in {".jsonl", ".ndjson"}:
        with open(p, "w", encoding="utf-8") as f:
            for _, row in df.iterrows():
                f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")
    else:
        raise ValueError("Unsupported output format. Use .csv or .jsonl")
