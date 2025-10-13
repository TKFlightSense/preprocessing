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
  --multi-label false
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
