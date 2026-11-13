import os
import random
import urllib.request

from datasets import load_dataset

SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/"
    "data/tinyshakespeare/input.txt"
)


def _clean(text: str) -> str:
    """Remove HTML line breaks found in IMDB reviews."""
    return text.replace("<br />", " ").strip()


def _validate_count(name: str, value: int) -> None:
    """Reject invalid sample sizes before querying or slicing a dataset."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer; got {value!r}")


def load_imdb_positive(n_train: int, n_val: int, seed: int):
    """Return (train_texts, val_texts): positive IMDB reviews for SFT."""
    _validate_count("n_train", n_train)
    _validate_count("n_val", n_val)
    ds = load_dataset("stanfordnlp/imdb", split="train")
    ds = ds.filter(lambda x: x["label"] == 1)  # keep positive reviews only
    requested = n_train + n_val
    if requested > len(ds):
        raise ValueError(
            f"Requested {requested} positive IMDB training reviews, "
            f"but the train split contains only {len(ds)}"
        )
    ds = ds.shuffle(seed=seed).select(range(n_train + n_val))
    texts = [_clean(t) for t in ds["text"]]
    return texts[:n_train], texts[n_train:]


def load_imdb_prompts(tokenizer, n_prompts: int, prompt_tokens: int, seed: int):
    """Return exactly sized prompts from the IMDB test split, in decoded text form."""
    _validate_count("n_prompts", n_prompts)
    _validate_count("prompt_tokens", prompt_tokens)
    ds = load_dataset("stanfordnlp/imdb", split="test")
    ds = ds.shuffle(seed=seed)
    prompts = []
    for text in ds["text"]:
        ids = tokenizer(
            _clean(text),
            add_special_tokens=False,
            truncation=True,
            max_length=prompt_tokens,
        )["input_ids"]
        if len(ids) < prompt_tokens:
            continue
        prompt = tokenizer.decode(
            ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        # Keep the string-to-token round trip at the requested length so every
        # evaluation and GRPO prompt starts from the same number of tokens.
        round_trip = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        if len(round_trip) == prompt_tokens:
            prompts.append(prompt)
        if len(prompts) == n_prompts:
            break

    if len(prompts) != n_prompts:
        raise ValueError(
            f"Could not build {n_prompts} IMDB prompts of {prompt_tokens} tokens; "
            f"built {len(prompts)}"
        )
    return prompts


def load_wikipedia(n_samples: int, seed: int):
    """Return Wikipedia paragraphs (WikiText-2 test split) used to measure perplexity."""
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    paragraphs = [
        t.strip() for t in ds["text"]
        if len(t.strip()) > 200 and not t.strip().startswith("=")  # skip titles
    ]
    random.Random(seed).shuffle(paragraphs)
    return paragraphs[:n_samples]


def load_shakespeare(n_samples: int, seed: int, cache_dir: str = "data"):
    """Return Shakespeare passages (Tiny Shakespeare), downloaded once and cached."""
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, "shakespeare.txt")
    if not os.path.exists(path):
        urllib.request.urlretrieve(SHAKESPEARE_URL, path)
    with open(path, encoding="utf-8") as f:
        text = f.read()
    passages = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 100]
    random.Random(seed).shuffle(passages)
    return passages[:n_samples]
