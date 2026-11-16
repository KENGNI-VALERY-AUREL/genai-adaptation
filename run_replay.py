"""SFT with Wikipedia replay: preserve general text capabilities."""

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset

from src.utils import load_config, set_seed, get_device
from src.model import load_model
from src.data import load_imdb_positive
from src.sft import train_sft


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-samples", type=int, default=5000)
    parser.add_argument("--replay-ratio", type=float, default=0.10)
    parser.add_argument("--name", default="sft_replay10")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    seed = cfg["seed"]
    set_seed(seed)

    if args.total_samples < 2:
        parser.error("--total-samples must be at least 2")
    if not 0 < args.replay_ratio < 1:
        parser.error("--replay-ratio must be between 0 and 1")
    if not args.name or Path(args.name).name != args.name:
        parser.error("--name must be a simple folder name")

    n_wiki = round(args.total_samples * args.replay_ratio)
    n_imdb = args.total_samples - n_wiki
    if n_wiki == 0 or n_imdb == 0:
        parser.error("Both data sources must contain at least one text")

    # Keep the existing frozen IMDb selection and validation set.
    imdb_train, val_texts = load_imdb_positive(
        cfg["data"]["n_train"],
        cfg["data"]["n_val"],
        seed,
    )
    if n_imdb > len(imdb_train):
        parser.error("Not enough frozen IMDb training examples")

    # Use TRAIN only. The project's WikiText TEST set stays untouched.
    wiki = load_dataset(
        "Salesforce/wikitext",
        "wikitext-2-raw-v1",
        split="train",
    )
    eligible = [
        i for i, text in enumerate(wiki["text"])
        if len(text.strip()) > 200
        and not text.strip().startswith("=")
    ]
    if n_wiki > len(eligible):
        parser.error("Not enough eligible Wikipedia paragraphs")

    # Select a deterministic set of Wikipedia rows.
    rng = random.Random(seed)
    wiki_indices = rng.sample(eligible, n_wiki)
    wiki_texts = [wiki[i]["text"].strip() for i in wiki_indices]

    train_texts = imdb_train[:n_imdb] + wiki_texts
    rng.shuffle(train_texts)

    print(f"IMDb training reviews: {n_imdb}")
    print(f"Wikipedia training paragraphs: {n_wiki}")
    print(f"Total training texts: {len(train_texts)}")
    print(f"IMDb validation reviews: {len(val_texts)}")

    # Inspect data preparation without loading or training GPT-2.
    if args.prepare_only:
        print("Preparation OK. No model loaded or trained.")
        return

    output_dir = Path(cfg["results_dir"]) / args.name
    checkpoint_dir = Path("checkpoints") / args.name
    if output_dir.exists() or checkpoint_dir.exists():
        raise FileExistsError(
            "Run name already exists. Choose another --name."
        )
    output_dir.mkdir(parents=True)

    cfg["sft"]["epochs"] = 1
    cfg["sft"]["checkpoint_dir"] = str(checkpoint_dir)

    metadata = {
        "seed": seed,
        "n_imdb": n_imdb,
        "n_wiki": n_wiki,
        "replay_ratio_by_examples": n_wiki / len(train_texts),
        "wiki_dataset": "Salesforce/wikitext",
        "wiki_config": "wikitext-2-raw-v1",
        "wiki_split": "train",
        "wiki_indices": wiki_indices,
        "config": cfg,
    }
    with (output_dir / "experiment.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(metadata, f, indent=2)

    device = get_device()
    print(f"Training device: {device}")
    model, tokenizer = load_model(cfg["model"]["name"], device)

    stats = train_sft(
        model, tokenizer, train_texts, val_texts, cfg, device,
        results_dir=str(output_dir),
        run_name=args.name,
    )
    with (output_dir / "cost.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"Checkpoint saved in: {checkpoint_dir}")
    print(f"Logs and configuration saved in: {output_dir}")


if __name__ == "__main__":
    main()