from src.utils import load_config
from src.model import load_model
from src.data import (
    load_imdb_positive, load_imdb_prompts, load_wikipedia, load_shakespeare,
)

cfg = load_config()
seed, d = cfg["seed"], cfg["data"]

_, tokenizer = load_model(cfg["model"]["name"], "cpu")

train, val = load_imdb_positive(d["n_train"], d["n_val"], seed)
prompts = load_imdb_prompts(tokenizer, d["n_test_prompts"], d["prompt_tokens"], seed)
wiki = load_wikipedia(d["n_wiki_samples"], seed)
shakes = load_shakespeare(d["n_shakespeare"], seed)

print(f"IMDB train: {len(train)} | val: {len(val)}")
print(f"IMDB test prompts: {len(prompts)} | example: {prompts[0]!r}")
print(f"Wikipedia: {len(wiki)} | example: {wiki[0][:100]!r}")
print(f"Shakespeare: {len(shakes)} | example: {shakes[0][:100]!r}")