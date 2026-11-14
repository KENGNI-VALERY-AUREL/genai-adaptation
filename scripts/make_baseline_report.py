"""Build the baseline table and figure from the saved results.

Usage (from the repository root):
    python scripts/make_baseline_report.py
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

RESULTS = "results"
RUNS = {
    "GPT-2 (raw)": {"eval": "eval_gpt2_raw.json", "cost": None},
    "SFT 1 epoch": {"eval": "eval_sft.json", "cost": "sft_cost.json"},
    "SFT 2 epochs": {"eval": "eval_sft_2ep.json", "cost": "sft_2ep_cost.json"},
    "SFT 3 epochs": {"eval": "eval_sft_3ep.json", "cost": "sft_3ep_cost.json"},
}
# The 3-epoch log contains the 2- and 1-epoch runs as prefixes (same seed, constant lr)
LOG_FILE = "sft_3ep_log.csv"

with open("configs/base.yaml") as f:
    cfg = yaml.safe_load(f)
# Number of generations behind each mean reward
n_samples = cfg["data"]["n_test_prompts"] * cfg["eval"]["n_generations"]


def load_json(name):
    """Load a JSON file from results/, or return None if it does not exist."""
    path = os.path.join(RESULTS, name)
    if name is None or not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------- table
rows = {}
for name, files in RUNS.items():
    res = load_json(files["eval"])
    cost = load_json(files["cost"]) if files["cost"] else None
    rows[name] = {
        "reward_mean": res["reward_mean"],
        # Standard error of the mean (optimistic: generations of one prompt are correlated)
        "reward_se": res["reward_std"] / np.sqrt(n_samples),
        "wiki_ppl": res["wiki_ppl"],
        "l2_distance": res["l2_distance"],
        "distinct_1": res["distinct_1"],
        "distinct_2": res["distinct_2"],
        "train_time_min": cost["train_time_s"] / 60 if cost else np.nan,
        "peak_gpu_gb": cost["peak_gpu_mem_gb"] if cost else np.nan,
    }
table = pd.DataFrame(rows).T.round(3)
table.to_csv(f"{RESULTS}/baseline_table.csv")
print(table)

with open(f"{RESULTS}/baseline_table.md", "w") as f:
    cols = list(table.columns)
    f.write("| Method | " + " | ".join(cols) + " |\n")
    f.write("|---" * (len(cols) + 1) + "|\n")
    for name, row in table.iterrows():
        f.write(f"| {name} | " + " | ".join("-" if pd.isna(v) else str(v)
                                             for v in row) + " |\n")

# ---------------------------------------------------------------- figure
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# Left: reward vs forgetting, with error bars (+-1 standard error)
ax = axes[0]
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
for (name, row), color in zip(table.iterrows(), colors):
    ax.errorbar(row["wiki_ppl"], row["reward_mean"], yerr=row["reward_se"],
                fmt="o", color=color, capsize=4, markersize=8)
    ax.annotate(name, (row["wiki_ppl"], row["reward_mean"]),
                textcoords="offset points", xytext=(8, 10), fontsize=9)
# Line linking the SFT runs: trajectory when training longer
sft = table.loc[["SFT 1 epoch", "SFT 2 epochs", "SFT 3 epochs"]]
ax.plot(sft["wiki_ppl"], sft["reward_mean"], "--", color="gray", alpha=0.6)
ax.set_xlabel("Wikipedia perplexity (lower = less forgetting)")
ax.set_ylabel("Mean reward (higher = more positive)")
ax.set_title("Reward vs. forgetting (error bars: ±1 SE)")
ax.set_xlim(table["wiki_ppl"].min() - 2, table["wiki_ppl"].max() + 5)
ax.set_ylim(table["reward_mean"].min() - 0.03, table["reward_mean"].max() + 0.04)
ax.grid(alpha=0.3)

# Right: full SFT training curve with the validation loss at each epoch end
ax = axes[1]
log = pd.read_csv(f"{RESULTS}/{LOG_FILE}")
train = log.dropna(subset=["train_loss"])
val = log.dropna(subset=["val_loss"])
ax.plot(train["step"], train["train_loss"], alpha=0.3, label="train loss (per step)")
ax.plot(train["step"], train["train_loss"].rolling(30).mean(),
        color="tab:orange", label="train loss (rolling mean)")
ax.plot(val["step"], val["val_loss"], "ks-", markersize=7, label="validation loss")
for step in val["step"].iloc[:-1]:
    ax.axvline(step, color="gray", linestyle=":", alpha=0.6)  # epoch boundaries
ax.set_xlabel("Step")
ax.set_ylabel("Loss")
ax.set_title("SFT training curve (3 epochs)")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{RESULTS}/baseline_plot.png", dpi=150)
print(f"Saved {RESULTS}/baseline_table.md and {RESULTS}/baseline_plot.png")