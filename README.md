# Labelicious

Minimal, strict labeling for training data. Given a CSV whose first column is `review` and a `labels.yaml` file, it produces a new CSV with two columns: `review,labels`. The model returns only the label text to minimize token usage and cost.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
cp .env.example .env  # set OPENAI_API_KEY
```

Environment (.env):
- `OPENAI_API_KEY=...`
- Optional: `OPENAI_BASE_URL=https://api.openai.com/v1`

## Quickstart (single-label, labels-only output)

```bash
labelicious classify \
  --input examples/sample_dataset.csv \
  --labels-file examples/labels.yaml \
  --output out.csv
```

- Input CSV: first column must be the review text (ideally named `review`). If it isn’t named, the first column is treated as `review` automatically.
- Labels YAML: a small, curated list under the `labels:` key (keep the set focused for quality/cost). Example: `examples/labels.yaml`.
- Output CSV: two columns with the original review and a single best label: `review,labels`.

Why this is cost‑efficient
- The prompt is strict and requests only the label string (no JSON, no rationale).
- One call per row, one label per row — ideal for training a BERT classifier.

## Input formats

- CSV (`.csv`): first column is used as `review`.
- Labels YAML:
  ```yaml
  labels:
    - Flight Delay / Cancellation
    - Customer Service
    - Baggage Issues
  ```

## Options

- `--model gpt-4o-mini` (default) and other OpenAI‑compatible models
- `--temperature 0 --top_p 1` for deterministic labeling (default in classify)
- `--max-rows N` to test on a subset

## Output

CSV with two columns:
- `review`: unchanged input text
- `labels`: the single best label, exactly as listed in `labels.yaml`
### Multi-label (optional)

If a review clearly fits more than one label, you can enable a strict multi-label mode that requests up to 3 labels, returned as a single field separated by `|` (pipe). This avoids CSV parsing ambiguity from commas.

```bash
labelicious classify \
  --input examples/sample_dataset.csv \
  --labels-file examples/labels.yaml \
  --output out_multi.csv \
  --multi-label
```

The output still has two columns: `review,labels`. In multi-label mode, the `labels` field can contain one, two, or three labels separated by `|`, e.g., `baggage_issues | inflight_experience | customer_service`.
