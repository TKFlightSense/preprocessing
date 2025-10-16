# Labelicious

Label unlabelled text with a list of candidate labels using a LLM API.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
cp .env.example .env  # set OPENAI_API_KEY
```

## Run

```bash
labelicious run \
  --input examples/sample_dataset.csv \
  --text-col text \
  --labels-file examples/labels.yaml \
  --output out.csv \
  --no-multi-label
```

### Run Options

```bash
labelicious run --input DATA.csv --text-col text \
  --labels-file labels.yaml --output labeled.csv \
  --provider openai --model gpt-4o-mini \
  --multi-label false --temperature 0 \
  --max-workers 8 --self-consistency 3
```

### Input

CSV with a text column (configurable)

YALM labels file

## Dual-label output (review, label_1, label_2)

To label each review with two independent label sets (e.g., domain category and intent):

```bash
labelicious run-dual \
  --input examples/sample_dataset.csv \
  --text-col text \
  --labels1-file examples/airline_labels.yaml \
  --labels2-file examples/sentiment_intent.yaml \
  --output out_dual.csv \
  --no-labels1-multi --no-labels2-multi
```

This writes a table with columns: `review,label_1,label_2`.

Single LLM call per row (faster, cheaper):

```bash
labelicious run-dual-single \
  --input examples/sample_dataset.csv \
  --text-col text \
  --labels1-file examples/airline_labels.yaml \
  --labels2-file examples/sentiment_intent.yaml \
  --output out_dual_single.csv \
  --no-labels1-multi --no-labels2-multi
```
