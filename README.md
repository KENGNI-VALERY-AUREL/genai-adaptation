# GenAI Adaptation

## Setup

The first step in your terminal is to install the project dependencies and requirements:

```bash
pip install -r requirements.txt
```

If you are using a virtual environment, activate it first and then run the command above from the project root.

## Project structure

- `src/` — source code
- `configs/` — configuration files
- `notebooks/` — notebooks for experimentation
- `results/` — generated outputs

## Frozen data selections

`configs/data_splits.json` is the versioned source of row indices for the IMDb
positive train/validation reviews, IMDb test prompts, WikiText reference
paragraphs, and Tiny Shakespeare passages. The loaders in `src/data.py` use
these indices directly, so changing the training seed does not silently change
which examples are used. IMDb train and validation indices are disjoint, and
the saved prompt indices were selected with the configured GPT-2 tokenizer.

To create the manifest for a new project setup, run this once and commit the
resulting JSON:

```bash
python scripts/freeze_splits.py
```

The script refuses to replace an existing manifest by default. Use `--overwrite`
only when the team intentionally agrees to create a new split version; review
the manifest diff before committing it. Normal training and evaluation runs
never regenerate the indices.

The current manifest freezes 5,000 IMDb training reviews, 500 validation
reviews, 200 test prompts of 8 GPT-2 tokens, 200 WikiText paragraphs, and 500
Shakespeare passages, matching `configs/base.yaml`.
