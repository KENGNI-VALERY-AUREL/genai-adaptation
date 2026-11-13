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


def load_imdb_positive(n_train: int, n_val: int, seed: int):
    """Return (train_texts, val_texts): positive IMDB reviews for SFT."""
    ds = load_dataset("stanfordnlp/imdb", split="train")
    ds = ds.filter(lambda x: x["label"] == 1)  # keep positive reviews only
    ds = ds.shuffle(seed=seed).select(range(n_train + n_val))
    texts = [_clean(t) for t in ds["text"]]
    return texts[:n_train], texts[n_train:]


def load_imdb_prompts(tokenizer, n_prompts: int, prompt_tokens: int, seed: int):
    """Return short prompts made of the first `prompt_tokens` tokens of test reviews."""
    ds = load_dataset("stanfordnlp/imdb", split="test")
    ds = ds.shuffle(seed=seed).select(range(n_prompts))
    prompts = []
    for text in ds["text"]:
        ids = tokenizer(_clean(text))["input_ids"][:prompt_tokens]
        prompts.append(tokenizer.decode(ids))
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