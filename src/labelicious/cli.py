from __future__ import annotations

from pathlib import Path
from typing import Optional
import difflib

import pandas as pd
import typer

from .schema import LabelInstruction, ModelConfig
from .dataset_io import load_labels_yaml, read_table, write_table
from .labeler import Labeler
from .prompts import labels_only_prompt, labels_only_prompt_multi, SYSTEM
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
    debug: bool = typer.Option(False, help="Print raw LLM outputs for debugging"),
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
    labeler = Labeler(li, cfg, debug=debug)
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
    debug: bool = typer.Option(False, help="Print raw LLM outputs for debugging"),
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

    labeler1 = Labeler(li1, cfg, debug=debug)
    labeler2 = Labeler(li2, cfg, debug=debug)

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
    debug: bool = typer.Option(False, help="Print raw LLM outputs for debugging"),
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
        if debug:
            print("[DEBUG][LLM][dual-single] raw:")
            print(res.content)
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


@app.command("classify")
def classify(
    input: Path = typer.Option(..., help="Input CSV/JSONL with first column 'review'"),
    labels_file: Path = typer.Option(..., help="YAML with labels: [labels: ...]"),
    output: Path = typer.Option(..., help="Output CSV/JSONL with columns: review,labels"),
    model: Optional[str] = typer.Option(None, help="Model name override"),
    provider: str = typer.Option("openai", help="LLM provider (only 'openai' supported)"),
    temperature: float = typer.Option(0.0, help="Sampling temperature (default 0 for determinism)"),
    top_p: float = typer.Option(1.0, help="Nucleus sampling top_p"),
    seed: Optional[int] = typer.Option(None, help="Random seed (if provider supports)"),
    max_rows: Optional[int] = typer.Option(None, help="Limit rows for a dry run"),
    multi_label: bool = typer.Option(False, help="Allow multiple labels (comma-separated, up to 3)"),
    debug: bool = typer.Option(False, help="Print raw LLM outputs for debugging"),
):
    """Simplified classification: review -> single label (labels-only output).

    Reads the first column 'review' from the input and writes two columns: review,labels.
    """
    if provider != "openai":
        raise typer.BadParameter("Currently only 'openai'-compatible provider is supported in classify mode")

    labels = load_labels_yaml(labels_file)
    allowed = labels
    allowed_norm = {lbl.lower().strip(): lbl for lbl in allowed}

    cfg = ModelConfig(provider=provider, model=model or "gpt-4o-mini")
    cfg.temperature = temperature
    cfg.top_p = top_p
    cfg.seed = seed

    client = OpenAIClient(model=cfg.model)

    df = read_table(input)
    # Ensure a 'review' column exists and is the first column
    if "review" not in df.columns:
        # If first column exists, rename it to 'review'
        if len(df.columns) >= 1:
            df = df.rename(columns={df.columns[0]: "review"})
        else:
            raise typer.BadParameter("Input file has no columns. Expected a 'review' column.")

    it = df.itertuples(index=False)
    if max_rows is not None:
        it = list(it)[:max_rows]

    rows = []
    for row in it:
        text = str(getattr(row, "review"))
        up = labels_only_prompt_multi(text, allowed) if multi_label else labels_only_prompt(text, allowed)
        res = client.chat(system=SYSTEM, user=up, temperature=cfg.temperature, top_p=cfg.top_p, seed=cfg.seed)
        if debug:
            print("[DEBUG][LLM][classify] raw:")
            print(res.content)
        content = (res.content or "").strip()
        # Strip code fences if any
        if content.startswith("```"):
            content = content.strip("`\n ")
            if content.lower().startswith("json"):
                content = content[4:].lstrip()
        # Take first line only to be safe
        content = content.splitlines()[0].strip()

        def map_one(tok: str) -> str | None:
            t = tok.strip().strip('"\'')
            if not t:
                return None
            lab = allowed_norm.get(t.lower())
            if lab:
                return lab
            best = difflib.get_close_matches(t.lower(), list(allowed_norm.keys()), n=1, cutoff=0.6)
            return allowed_norm[best[0]] if best else None

        labels_out: list[str] = []
        if multi_label:
            # Split on commas or newlines
            parts = [p for p in [s.strip() for s in content.replace("\n", ",").split(",")] if p]
            for p in parts:
                m = map_one(p)
                if m and m not in labels_out:
                    labels_out.append(m)
                if len(labels_out) >= 3:
                    break
            if not labels_out:
                # fallback to first label
                labels_out = [allowed[0]]
            rows.append({"review": text, "labels": " | ".join(labels_out)})
        else:
            m = map_one(content)
            label = m if m else allowed[0]
            rows.append({"review": text, "labels": label})

    out_df = pd.DataFrame(rows, columns=["review", "labels"])
    write_table(out_df, output)
    typer.secho(f"Wrote {len(out_df)} rows to {output}", fg=typer.colors.GREEN)


@app.command("segment-dual")
def segment_dual(
    input: Path = typer.Option(..., help="Input CSV/JSONL"),
    text_col: str = typer.Option("text", help="Column containing the review text"),
    labels1_file: Path = typer.Option(..., help="YAML with primary labels (label_1)"),
    labels2_file: Path = typer.Option(..., help="YAML with secondary labels (label_2)"),
    output: Path = typer.Option(..., help="Output CSV/JSONL with segmented rows: review,label_1,label_2"),
    model: Optional[str] = typer.Option(None, help="Model name override"),
    provider: str = typer.Option("openai", help="LLM provider (only 'openai' supported)"),
    temperature: float = typer.Option(None, help="Override sampling temperature"),
    top_p: float = typer.Option(None, help="Override nucleus sampling top_p"),
    seed: Optional[int] = typer.Option(None, help="Random seed for determinism (if provider supports)"),
    max_rows: Optional[int] = typer.Option(None, help="Limit rows for a dry run"),
    debug: bool = typer.Option(False, help="Print raw LLM outputs for debugging"),
):
    """Segment reviews by primary label and also assign secondary label per segment.

    Produces one row per segment with columns: review (exact span), label_1, label_2.
    """
    if provider != "openai":
        raise typer.BadParameter("Currently only 'openai'-compatible provider is supported in segment-dual mode")

    labels1 = load_labels_yaml(labels1_file)
    labels2 = load_labels_yaml(labels2_file)
    allowed1 = set(labels1)
    allowed2 = set(labels2)

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
        full_text = str(getattr(row, text_col))
        up = segmentation_user_prompt(full_text, labels1, labels2)
        res = client.chat(system=SYSTEM, user=up, temperature=cfg.temperature, top_p=cfg.top_p, seed=cfg.seed)
        if debug:
            print("[DEBUG][LLM][segment] raw:")
            print(res.content)
        content = res.content.strip()
        if content.startswith("```"):
            content = content.strip("`\n ")
            if content.lower().startswith("json"):
                content = content[4:].lstrip()

        import json as _json
        try:
            data = _json.loads(content)
        except Exception as e:
            raise typer.BadParameter(f"Model returned non-JSON response: {content[:120]}... ({e})")

        segs = data.get("segments") or []
        if not isinstance(segs, list):
            segs = []

        n = len(full_text)
        for s in segs:
            try:
                start = int(s.get("start"))
                end = int(s.get("end"))
                l1 = str(s.get("label_1")) if s.get("label_1") is not None else ""
                l2 = str(s.get("label_2")) if s.get("label_2") is not None else ""
            except Exception:
                continue
            # validate bounds and labels
            if end <= start:
                continue
            if start < 0 or end > n:
                # clamp safely
                start = max(0, start)
                end = min(n, end)
                if end <= start:
                    continue
            if l1 not in allowed1 or l2 not in allowed2:
                continue
            snippet = full_text[start:end]
            # Skip very short spans
            if len(snippet.strip()) < 3:
                continue
            rows.append({"review": snippet, "label_1": l1, "label_2": l2})

        # Fallback: if no segments returned, create one whole-text segment with best single labels via dual-single
        if not rows:
            up2 = dual_user_prompt(full_text, labels1, labels2, multi1=False, multi2=False)
            res2 = client.chat(system=SYSTEM, user=up2, temperature=cfg.temperature, top_p=cfg.top_p, seed=cfg.seed)
            if debug:
                print("[DEBUG][LLM][segment-fallback] raw:")
                print(res2.content)
            c2 = res2.content.strip()
            if c2.startswith("```"):
                c2 = c2.strip("`\n ")
                if c2.lower().startswith("json"):
                    c2 = c2[4:].lstrip()
            try:
                d2 = _json.loads(c2)
                l1s = d2.get("label_1") or []
                l2s = d2.get("label_2") or []
                l1_val = l1s[0] if isinstance(l1s, list) and l1s else ""
                l2_val = l2s[0] if isinstance(l2s, list) and l2s else ""
                if l1_val in allowed1 and l2_val in allowed2:
                    rows.append({"review": full_text, "label_1": l1_val, "label_2": l2_val})
            except Exception:
                # ignore fallback errors
                pass

    out_df = pd.DataFrame(rows, columns=["review", "label_1", "label_2"])
    write_table(out_df, output)
    typer.secho(f"Wrote {len(out_df)} rows to {output}", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()
