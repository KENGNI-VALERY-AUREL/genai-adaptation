# Wikipedia Replay Experiment

## Objective

Test whether mixing general-domain text into SFT training reduces
degradation on WikiText while maintaining positive movie-review generation.

## Method

Both configurations start from the original GPT-2 and use one epoch,
learning rate 5e-5, batch size 16, maximum sequence length 128, and seed 42.

- Baseline: 5,000 positive IMDb reviews.
- Replay: 4,500 positive IMDb reviews and 500 WikiText-2 training paragraphs.
- The same 500 IMDb validation reviews are used for both configurations.
- Wikipedia replay paragraphs come only from the training split.
- Selected Wikipedia row indices are recorded in
  results/sft_replay10/experiment.json.

Both models were trained and evaluated in the same Colab T4 environment.
Evaluation uses the fixed 200 IMDb test prompts, four generations per
prompt, and 200 WikiText-2 test paragraphs.

## Results

| Metric | SFT baseline | SFT + 10% replay |
|---|---:|---:|
| Mean positivity reward | 0.687049 | 0.719449 |
| WikiText perplexity | 53.655106 | 36.394939 |
| L2 weight distance | 24.397799 | 21.978918 |
| Distinct-1 | 0.121059 | 0.123934 |
| Distinct-2 | 0.470038 | 0.474142 |
| Training time | About 2.9 min | About 2.9 min |
| Peak GPU memory | About 5.09 GB | About 5.09 GB |

Replay reduces WikiText perplexity by approximately 32.2% relative
to the SFT baseline. Mean positivity reward increases by 0.0324.

## Limitations

- These are preliminary results from one seed, without a significance test.
- The sentiment classifier is an imperfect proxy for generation quality.
- Reward and diversity are computed on prompts plus continuations.
- Replay training and perplexity evaluation use different splits of the
  same WikiText dataset; broader generalization has not been established.
- Replay replaces 500 IMDb reviews, so both data composition and the amount
  of task-specific training change.
- Equal numbers of training texts do not guarantee equal token budgets.
- The script performs model adaptation, not inference acceleration.

## Reproduction

Run from the project root in an environment with the project dependencies:

```bash
python run_replay.py --prepare-only
python run_replay.py --name sft_replay10
python run_eval.py --checkpoint checkpoints/sft_replay10 --name sft_replay10
python run_sft.py
python run_eval.py --checkpoint checkpoints/sft --name sft_baseline_same_env
```

The baseline configuration must specify one SFT epoch.
Use a new replay run name if its output folders already exist.
Baseline training writes to the checkpoint directory in configs/base.yaml.

The Colab package versions are recorded in results/farah_environment.txt.
Model checkpoints are stored separately and are not committed to Git.