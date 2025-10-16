from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from .schema import LabelInstruction, ModelConfig
from .dataset_io import load_labels_yaml, read_table, write_table
from .labeler import Labeler
from .prompts import dual_user_prompt, SYSTEM
from .llm_client import OpenAIClient

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


@app.command("run-dual")
def run_dual(
    input: Path = typer.Option(..., help="Input CSV/JSONL"),
    text_col: str = typer.Option("text", help="Column containing the review text"),
    labels1_file: Path = typer.Option(..., help="YAML with primary labels (label_1)"),
    labels2_file: Path = typer.Option(..., help="YAML with secondary labels (label_2)"),
    output: Path = typer.Option(..., help="Output CSV/JSONL with columns: review,label_1,label_2"),
    model: Optional[str] = typer.Option(None, help="Model name override"),
    provider: str = typer.Option("openai", help="LLM provider"),
    temperature: float = typer.Option(None, help="Override sampling temperature"),
    top_p: float = typer.Option(None, help="Override nucleus sampling top_p"),
    self_consistency: int = typer.Option(1, help="# independent samples for voting/aggregation"),
    seed: Optional[int] = typer.Option(None, help="Random seed for determinism (if provider supports)"),
    max_rows: Optional[int] = typer.Option(None, help="Limit rows for a dry run"),
    labels1_multi: bool = typer.Option(False, help="Allow multiple labels for label_1"),
    labels2_multi: bool = typer.Option(False, help="Allow multiple labels for label_2"),
):
    """Label each review into two label sets and write review,label_1,label_2."""
    # Load label sets
    labels1 = load_labels_yaml(labels1_file)
    labels2 = load_labels_yaml(labels2_file)

    li1 = LabelInstruction(labels=labels1, multi_label=labels1_multi, require_confidence=False)
    li2 = LabelInstruction(labels=labels2, multi_label=labels2_multi, require_confidence=False)

    cfg = ModelConfig(provider=provider, model=model or "gpt-4o-mini")
    if temperature is not None:
        cfg.temperature = temperature
    if top_p is not None:
        cfg.top_p = top_p
    cfg.seed = seed

    df = read_table(input)
    if text_col not in df.columns:
        raise typer.BadParameter(f"Text column '{text_col}' not found. Columns: {list(df.columns)}")

    labeler1 = Labeler(li1, cfg)
    labeler2 = Labeler(li2, cfg)

    # Apply optional row limit
    it = df.itertuples(index=False)
    if max_rows is not None:
        it = list(it)[:max_rows]

    rows = []
    for row in it:
        text = str(getattr(row, text_col))
        res1 = labeler1.label_text(text, self_consistency=self_consistency)
        res2 = labeler2.label_text(text, self_consistency=self_consistency)

        def to_cell(r):
            labs = r.response.predicted_labels or []
            return ", ".join(labs) if labs else ""

        rows.append({
            "review": text,
            "label_1": to_cell(res1),
            "label_2": to_cell(res2),
        })

    out_df = pd.DataFrame(rows, columns=["review", "label_1", "label_2"])
    write_table(out_df, output)
    typer.secho(f"Wrote {len(out_df)} rows to {output}", fg=typer.colors.GREEN)


@app.command("run-dual-single")
def run_dual_single(
    input: Path = typer.Option(..., help="Input CSV/JSONL"),
    text_col: str = typer.Option("text", help="Column containing the review text"),
    labels1_file: Path = typer.Option(..., help="YAML with primary labels (label_1)"),
    labels2_file: Path = typer.Option(..., help="YAML with secondary labels (label_2)"),
    output: Path = typer.Option(..., help="Output CSV/JSONL with columns: review,label_1,label_2"),
    model: Optional[str] = typer.Option(None, help="Model name override"),
    provider: str = typer.Option("openai", help="LLM provider (only 'openai' supported)"),
    temperature: float = typer.Option(None, help="Override sampling temperature"),
    top_p: float = typer.Option(None, help="Override nucleus sampling top_p"),
    seed: Optional[int] = typer.Option(None, help="Random seed for determinism (if provider supports)"),
    max_rows: Optional[int] = typer.Option(None, help="Limit rows for a dry run"),
    labels1_multi: bool = typer.Option(False, help="Allow multiple labels for label_1"),
    labels2_multi: bool = typer.Option(False, help="Allow multiple labels for label_2"),
):
    """Label each review into two label sets using a single LLM call per row."""
    if provider != "openai":
        raise typer.BadParameter("Currently only 'openai'-compatible provider is supported in single-call mode")

    labels1 = load_labels_yaml(labels1_file)
    labels2 = load_labels_yaml(labels2_file)

    # Model config and client
    cfg = ModelConfig(provider=provider, model=model or "gpt-4o-mini")
    if temperature is not None:
        cfg.temperature = temperature
    if top_p is not None:
        cfg.top_p = top_p
    cfg.seed = seed

    client = OpenAIClient(model=cfg.model)

    df = read_table(input)
    if text_col not in df.columns:
        raise typer.BadParameter(f"Text column '{text_col}' not found. Columns: {list(df.columns)}")

    it = df.itertuples(index=False)
    if max_rows is not None:
        it = list(it)[:max_rows]

    rows = []
    for row in it:
        text = str(getattr(row, text_col))
        up = dual_user_prompt(text, labels1, labels2, labels1_multi, labels2_multi)
        res = client.chat(system=SYSTEM, user=up, temperature=cfg.temperature, top_p=cfg.top_p, seed=cfg.seed)
        content = res.content.strip()
        if content.startswith("```"):
            content = content.strip("`\n ")
            if content.lower().startswith("json"):
                content = content[4:].lstrip()

        # Parse JSON and coerce shapes
        import json as _json
        try:
            data = _json.loads(content)
        except Exception as e:
            raise typer.BadParameter(f"Model returned non-JSON response: {content[:120]}... ({e})")

        def to_list(val):
            if val is None:
                return []
            if isinstance(val, list):
                return [str(x) for x in val]
            return [str(val)]

        l1 = to_list(data.get("label_1"))
        l2 = to_list(data.get("label_2"))
        if not labels1_multi and l1:
            l1 = [l1[0]]
        if not labels2_multi and l2:
            l2 = [l2[0]]

        rows.append({
            "review": text,
            "label_1": ", ".join(l1),
            "label_2": ", ".join(l2),
        })

    out_df = pd.DataFrame(rows, columns=["review", "label_1", "label_2"])
    write_table(out_df, output)
    typer.secho(f"Wrote {len(out_df)} rows to {output}", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()
