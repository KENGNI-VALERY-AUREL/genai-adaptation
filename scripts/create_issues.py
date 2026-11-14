"""Create the Sprint 1 labels, milestone and issues on GitHub (requires GitHub CLI).

Usage (from the repository root):
    python scripts/create_issues.py --dry-run   # preview only
    python scripts/create_issues.py             # create everything (run ONCE)
"""
import subprocess
import sys

DRY_RUN = "--dry-run" in sys.argv
MILESTONE = "Sprint 1"
# Tasks already completed: their issues are created and then closed
DONE = {"A1", "A3", "A4", "A5", "B1", "B2", "C1", "C2", "C3",
        "D1", "D2", "E1", "F2", "F3"}

LABELS = {
    "sprint-1": ("0E8A16", "Sprint 1 task"),
    "A-setup": ("1D76DB", "Setup and organization"),
    "B-model": ("5319E7", "Model"),
    "C-data": ("FBCA04", "Data"),
    "D-reward": ("D93F0B", "Reward model"),
    "E-sft": ("0052CC", "SFT baseline"),
    "F-evaluation": ("B60205", "Evaluation protocol"),
    "G-results": ("006B75", "Results and deliverable"),
}

# (id, label, title, description, done when, depends on)
TASKS = [
    ("A1", "A-setup", "Initialize the Git repository",
     "Structure src/, configs/, notebooks/, results/, .gitignore, minimal README.",
     "All three members have cloned the repository and can push.", "-"),
    ("A2", "A-setup", "Define working conventions",
     "Branches, pull requests, issue naming.",
     "The convention is written in the README.", "A1"),
    ("A3", "A-setup", "Create a reproducible environment",
     "requirements.txt with pinned versions (torch, transformers, datasets).",
     "A clean install works on every member's machine.", "A1"),
    ("A4", "A-setup", "Fix seeds and config",
     "set_seed() function, hyperparameters in YAML files.",
     "Two identical runs give the same results.", "A3"),
    ("A5", "A-setup", "Set up logging",
     "Save metrics as CSV (or W&B) in results/.",
     "A test training run writes its log.", "A4"),
    ("A6", "A-setup", "Choose the compute platform",
     "Local CPU is too slow for full training: decide on Colab or a member's GPU and document how to run the code.",
     "Everyone can run a test script.", "A3"),
    ("B1", "B-model", "Load GPT-2 124M",
     "HuggingFace loading in src/model.py.",
     "Loading works on CPU and GPU.", "A3, A4"),
    ("B2", "B-model", "Write the generation function",
     "Prompt, number of tokens, sampling (temperature, top-k).",
     "GPT-2 generates a review continuation from an 8-token prompt.", "B1"),
    ("C1", "C-data", "Load and prepare IMDB",
     "8-token prompts for evaluation and GRPO, positive reviews for SFT.",
     "The data loads and tokenizes without errors.", "A3, B1"),
    ("C2", "C-data", "Prepare the Wikipedia reference sample",
     "Neutral text used to measure perplexity and forgetting.",
     "The sample is saved and its size documented.", "A3"),
    ("C3", "C-data", "Prepare Shakespeare",
     "Needed for the Sprint 2 task vectors.",
     "The dataset is loaded and split.", "A3"),
    ("C4", "C-data", "Freeze train/val/test splits",
     "Save the indices so everyone uses the same ones.",
     "The splits are versioned and no longer change.", "C1, C2, C3"),
    ("D1", "D-reward", "Integrate the sentiment classifier",
     "DistilBERT SST-2 in src/reward.py.",
     "reward(texts) returns P(positive) for a list of texts.", "A3"),
    ("D2", "D-reward", "Test the reward model on edge cases",
     "Very positive, very negative, repetitive text ('great great great'), off-topic text.",
     "A small score table is produced and weaknesses are noted.", "D1"),
    ("E1", "E-sft", "Implement the SFT loop",
     "Language modeling loss on positive reviews, optimizer, batches.",
     "Training runs end to end.", "B1, C1, C4"),
    ("E2", "E-sft", "Run the baseline training",
     "Loss curves and saved checkpoints.",
     "A final checkpoint and its curves are in results/.", "E1, A5"),
    ("F1", "F-evaluation", "Fix the test size",
     "Number of prompts and generations per prompt.",
     "The values are in the config and will not change.", "C4"),
    ("F2", "F-evaluation", "Implement the metrics",
     "Mean reward, Wikipedia perplexity, L2 weight distance from initial GPT-2, diversity (distinct-n).",
     "Each metric is tested on raw GPT-2.", "B1, C2, D1"),
    ("F3", "F-evaluation", "Write evaluate(model)",
     "A single function returning all metrics, reused by every Sprint 2 method.",
     "It runs on any checkpoint.", "F1, F2"),
    ("F4", "F-evaluation", "Measure cost",
     "Training time and GPU/RAM memory.",
     "These values are logged for SFT.", "E2"),
    ("G1", "G-results", "Evaluate raw GPT-2 and SFT",
     "Same protocol for both.",
     "The raw results are saved.", "F3, E2"),
    ("G2", "G-results", "Produce the baseline table and a first plot",
     "GPT-2 vs. SFT on all metrics.",
     "Table and figure are exported.", "G1"),
    ("G3", "G-results", "Record observations",
     "Example generations, first signs of forgetting or drift.",
     "A short note is written.", "G1"),
    ("G4", "G-results", "Prepare 1-2 slides for the Week 10 meeting",
     "Baseline table, plot and key observations.",
     "The slides are ready.", "G2"),
    ("G5", "G-results", "Update the README to reproduce Sprint 1",
     "Install and run commands for every Sprint 1 step.",
     "An outsider can rerun the baseline.", "G1"),
]


def run(cmd, check=True):
    """Run a gh command (or only print it in dry-run mode) and return its output."""
    print("->", " ".join(cmd[:5]), "...")
    if DRY_RUN:
        return ""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if check and result.returncode != 0:
        sys.exit(f"Command failed:\n{result.stderr}")
    return result.stdout.strip()


def main():
    for name, (color, desc) in LABELS.items():
        run(["gh", "label", "create", name, "--color", color,
             "--description", desc, "--force"])

    # May fail harmlessly if the milestone already exists
    run(["gh", "api", "repos/{owner}/{repo}/milestones",
         "-f", f"title={MILESTONE}",
         "-f", "description=Foundations and SFT baseline (W6-W9)"], check=False)

    for tid, label, title, desc, done_when, depends in TASKS:
        body = (f"{desc}\n\n**Done when:** {done_when}\n\n**Depends on:** {depends}")
        url = run(["gh", "issue", "create",
                   "--title", f"[{tid}] {title}",
                   "--body", body,
                   "--label", f"sprint-1,{label}",
                   "--milestone", MILESTONE])
        if tid in DONE:
            run(["gh", "issue", "close", url, "--reason", "completed"])

    print("Done.")


if __name__ == "__main__":
    main()