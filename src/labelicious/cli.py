from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from .schema import LabelInstruction, ModelConfig
from .dataset_io import load_labels_yaml, read_table, write_table
from .labeler import Labeler

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def validate(labels_file: Path):
    """
    Validate labels YAML file structure.
    """
    labels = load_labels_yaml(labels_file)
    _ = LabelInstruction(labels=labels)
    typer.secho(f"OK: {len(labels)} labels", fg=typer.colors.GREEN)


@app.command()
def run(
    input: Path = typer.Option(..., help="Input CSV/JSONL"),
    text_col: str = typer.Option("text", help="Column containing text to classify"),
    labels_file: Path = typer.Option(..., help="YAML file with labels: [labels: ...]"),
    output: Path = typer.Option(..., help="Output CSV/JSONL"),
    multi_label: bool = typer.Option(False, help="Allow multiple labels per row"),
    temperature: float = typer.Option(None, help="Override sampling temperature"),
    top_p: float = typer.Option(None, help="Override nucleus sampling top_p"),
    model: Optional[str] = typer.Option(None, help="Model name override"),
    provider: str = typer.Option("openai", help="LLM provider"),
    self_consistency: int = typer.Option(1, help="# independent samples for voting/aggregation"),
    seed: Optional[int] = typer.Option(None, help="Random seed for determinism (if provider supports)"),
    max_rows: Optional[int] = typer.Option(None, help="Limit rows for a dry run"),
):
    """
    Label a dataset with a provided label set using an LLM.
    """
    labels = load_labels_yaml(labels_file)
    li = LabelInstruction(labels=labels, multi_label=multi_label, require_confidence=True)

    cfg = ModelConfig(provider=provider, model=model or "gpt-4o-mini")
    if temperature is not None:
        cfg.temperature = temperature
    if top_p is not None:
        cfg.top_p = top_p
    cfg.seed = seed

    df = read_table(input)
    labeler = Labeler(li, cfg)
    out_df = labeler.label_dataframe(
        df, text_col=text_col, self_consistency=self_consistency, max_rows=max_rows
    )

    write_table(out_df, output)
    typer.secho(f"Wrote {len(out_df)} rows to {output}", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()
