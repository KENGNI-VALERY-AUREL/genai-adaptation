"""Generate the versioned dataset row selections used by all experiments.

Run from the repository root once when creating a new split manifest:
    python scripts/freeze_splits.py

The script refuses to replace an existing manifest unless --overwrite is passed.
Review and commit configs/data_splits.json whenever intentionally regenerating it.
"""
import argparse
import json
import random
import sys
import urllib.request
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import SHAKESPEARE_URL, _clean  # noqa: E402
from src.utils import load_config  # noqa: E402


def _with_source_indices(dataset):
    return dataset.add_column("_source_index", list(range(len(dataset))))


def _imdb_splits(config, tokenizer):
    seed, data_cfg = config["seed"], config["data"]
    train = _with_source_indices(load_dataset("stanfordnlp/imdb", split="train"))
    positive = train.filter(lambda row: row["label"] == 1)
    requested = data_cfg["n_train"] + data_cfg["n_val"]
    if requested > len(positive):
        raise ValueError(f"Requested {requested} positive IMDb reviews, found {len(positive)}")
    selected = positive.shuffle(seed=seed).select(range(requested))
    source_indices = selected["_source_index"]

    test = _with_source_indices(load_dataset("stanfordnlp/imdb", split="test"))
    shuffled_test = test.shuffle(seed=seed)
    prompt_indices = []
    for source_index, text in zip(shuffled_test["_source_index"], shuffled_test["text"]):
        ids = tokenizer(
            _clean(text),
            add_special_tokens=False,
            truncation=True,
            max_length=data_cfg["prompt_tokens"],
        )["input_ids"]
        if len(ids) < data_cfg["prompt_tokens"]:
            continue
        prompt = tokenizer.decode(
            ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if len(tokenizer(prompt, add_special_tokens=False)["input_ids"]) == data_cfg["prompt_tokens"]:
            prompt_indices.append(source_index)
        if len(prompt_indices) == data_cfg["n_test_prompts"]:
            break
    if len(prompt_indices) != data_cfg["n_test_prompts"]:
        raise ValueError(
            f"Could not freeze {data_cfg['n_test_prompts']} IMDb prompts of "
            f"{data_cfg['prompt_tokens']} tokens"
        )

    return {
        "train_positive_indices": source_indices[:data_cfg["n_train"]],
        "validation_positive_indices": source_indices[data_cfg["n_train"]:],
        "test_prompt_indices": prompt_indices,
    }


def _wikipedia_indices(config):
    sample_count = config["data"]["n_wiki_samples"]
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    eligible = [
        (i, text.strip())
        for i, text in enumerate(ds["text"])
        if len(text.strip()) > 200 and not text.strip().startswith("=")
    ]
    random.Random(config["seed"]).shuffle(eligible)
    if sample_count > len(eligible):
        raise ValueError(f"Requested {sample_count} WikiText paragraphs, found {len(eligible)}")
    return [i for i, _ in eligible[:sample_count]]


def _shakespeare_indices(config):
    cache_dir = ROOT / "data"
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "shakespeare.txt"
    if not path.exists():
        urllib.request.urlretrieve(SHAKESPEARE_URL, path)
    text = path.read_text(encoding="utf-8")
    passages = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 100]
    indexed_passages = list(enumerate(passages))
    random.Random(config["seed"]).shuffle(indexed_passages)
    sample_count = config["data"]["n_shakespeare"]
    if sample_count > len(indexed_passages):
        raise ValueError(f"Requested {sample_count} Shakespeare passages, found {len(indexed_passages)}")
    return [i for i, _ in indexed_passages[:sample_count]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true", help="replace an existing frozen manifest")
    args = parser.parse_args()

    output = ROOT / "configs" / "data_splits.json"
    if output.exists() and not args.overwrite:
        raise SystemExit(f"{output} already exists; pass --overwrite only to intentionally replace it")

    config = load_config(str(ROOT / "configs" / "base.yaml"))
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])
    splits = {
        "version": 1,
        "seed_used_to_create": config["seed"],
        "sources": {
            "imdb": "stanfordnlp/imdb (train and test splits; indices refer to source rows)",
            "wikipedia": "Salesforce/wikitext, wikitext-2-raw-v1 (test split; indices refer to source rows)",
            "shakespeare": SHAKESPEARE_URL + " (indices refer to eligible passages after filtering)",
            "prompt_tokenizer": config["model"]["name"],
        },
        "imdb": _imdb_splits(config, tokenizer),
        "wikipedia": {"test_paragraph_indices": _wikipedia_indices(config)},
        "shakespeare": {"passage_indices": _shakespeare_indices(config)},
    }
    output.write_text(json.dumps(splits, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote frozen splits to {output}")
    print(
        "Counts: "
        f"IMDb train={len(splits['imdb']['train_positive_indices'])}, "
        f"validation={len(splits['imdb']['validation_positive_indices'])}, "
        f"prompts={len(splits['imdb']['test_prompt_indices'])}; "
        f"Wikipedia={len(splits['wikipedia']['test_paragraph_indices'])}; "
        f"Shakespeare={len(splits['shakespeare']['passage_indices'])}"
    )


if __name__ == "__main__":
    main()
