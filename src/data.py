import json
import os
import urllib.request
from pathlib import Path

from datasets import load_dataset

SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/"
    "data/tinyshakespeare/input.txt"
)
SPLITS_PATH = Path(__file__).resolve().parents[1] / "configs" / "data_splits.json"


def _clean(text: str) -> str:
    """Remove HTML line breaks found in IMDB reviews."""
    return text.replace("<br />", " ").strip()


def _validate_count(name: str, value: int) -> None:
    """Reject invalid sample sizes before querying or slicing a dataset."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer; got {value!r}")


def _frozen_indices(section: str, name: str, count: int) -> list[int]:
    """Read a fixed, version-controlled index list from the split manifest."""
    _validate_count("count", count)
    try:
        manifest = json.loads(SPLITS_PATH.read_text(encoding="utf-8"))
        indices = manifest[section][name]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise RuntimeError(
            f"Could not read frozen split '{section}.{name}' from {SPLITS_PATH}"
        ) from exc

    if not isinstance(indices, list) or any(
        isinstance(index, bool) or not isinstance(index, int) or index < 0
        for index in indices
    ):
        raise ValueError(f"Frozen split '{section}.{name}' must be a list of nonnegative integers")
    if len(set(indices)) != len(indices):
        raise ValueError(f"Frozen split '{section}.{name}' contains duplicate indices")
    if len(indices) < count:
        raise ValueError(
            f"Frozen split '{section}.{name}' contains {len(indices)} indices, "
            f"but {count} were requested"
        )
    return indices[:count]


def _select_rows(dataset, indices: list[int], split_name: str):
    """Select frozen row indices, failing clearly if the source split changed."""
    invalid = [index for index in indices if index >= len(dataset)]
    if invalid:
        raise ValueError(
            f"Frozen indices for {split_name} include row {max(invalid)}, "
            f"but the source split has only {len(dataset)} rows"
        )
    if hasattr(dataset, "select"):
        return dataset.select(indices)
    return [dataset[index] for index in indices]


def load_imdb_positive(n_train: int, n_val: int, seed: int):
    """Return the frozen positive IMDb train and validation reviews for SFT.

    ``seed`` remains in the signature for compatibility; membership is fixed by
    ``configs/data_splits.json`` rather than shuffled at load time.
    """
    _validate_count("n_train", n_train)
    _validate_count("n_val", n_val)
    train_indices = _frozen_indices("imdb", "train_positive_indices", n_train)
    val_indices = _frozen_indices("imdb", "validation_positive_indices", n_val)
    if set(train_indices).intersection(val_indices):
        raise ValueError("Frozen IMDb train and validation splits overlap")
    ds = load_dataset("stanfordnlp/imdb", split="train")
    train = _select_rows(ds, train_indices, "IMDb train")
    val = _select_rows(ds, val_indices, "IMDb validation")
    if (any(label != 1 for label in train["label"])
            or any(label != 1 for label in val["label"])):
        raise ValueError("Frozen IMDb train/validation indices must refer only to positive reviews")
    return ([_clean(t) for t in train["text"]],
            [_clean(t) for t in val["text"]])


def load_imdb_prompts(tokenizer, n_prompts: int, prompt_tokens: int, seed: int):
    """Return the frozen, exactly sized IMDb test prompts in decoded text form."""
    _validate_count("n_prompts", n_prompts)
    _validate_count("prompt_tokens", prompt_tokens)
    prompt_indices = _frozen_indices("imdb", "test_prompt_indices", n_prompts)
    ds = _select_rows(
        load_dataset("stanfordnlp/imdb", split="test"),
        prompt_indices,
        "IMDb test",
    )
    prompts = []
    for source_index, text in zip(prompt_indices, ds["text"]):
        ids = tokenizer(
            _clean(text),
            add_special_tokens=False,
            truncation=True,
            max_length=prompt_tokens,
        )["input_ids"]
        if len(ids) < prompt_tokens:
            raise ValueError(
                f"Frozen IMDb test row {source_index} is too short for "
                f"a {prompt_tokens}-token prompt"
            )
        prompt = tokenizer.decode(
            ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        # Keep the string-to-token round trip at the requested length so every
        # evaluation and GRPO prompt starts from the same number of tokens.
        round_trip = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        if len(round_trip) != prompt_tokens:
            raise ValueError(
                f"Frozen IMDb test row {source_index} does not round-trip to "
                f"exactly {prompt_tokens} tokens"
            )
        prompts.append(prompt)
    return prompts


def load_wikipedia(n_samples: int, seed: int):
    """Return the frozen WikiText-2 reference paragraphs used for perplexity."""
    indices = _frozen_indices("wikipedia", "test_paragraph_indices", n_samples)
    ds = _select_rows(
        load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test"),
        indices,
        "WikiText-2 test",
    )
    paragraphs = [text.strip() for text in ds["text"]]
    if any(len(text) <= 200 or text.startswith("=") for text in paragraphs):
        raise ValueError("Frozen WikiText-2 indices must refer to non-title paragraphs over 200 chars")
    return paragraphs


def load_shakespeare(n_samples: int, seed: int, cache_dir: str = "data"):
    """Return the frozen Tiny Shakespeare passages, downloaded once and cached."""
    indices = _frozen_indices("shakespeare", "passage_indices", n_samples)
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, "shakespeare.txt")
    if not os.path.exists(path):
        urllib.request.urlretrieve(SHAKESPEARE_URL, path)
    with open(path, encoding="utf-8") as f:
        text = f.read()
    passages = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 100]
    return _select_rows(passages, indices, "Tiny Shakespeare passages")
