import argparse
import json
import os

from src.utils import set_seed, get_device, load_config
from src.model import load_model
from src.data import load_imdb_prompts, load_wikipedia
from src.reward import RewardModel
from src.evaluate import evaluate, get_reference_state

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default=None,
                    help="path to a saved model (default: raw GPT-2)")
parser.add_argument("--name", default="gpt2_raw", help="name of the result file")
parser.add_argument("--max-prompts", type=int, default=None,
                    help="use only N test prompts (quick test)")
args = parser.parse_args()

cfg = load_config()
set_seed(cfg["seed"])
device = get_device()
d = cfg["data"]

# The initial GPT-2 is always the reference for the L2 distance
base_model, tokenizer = load_model(cfg["model"]["name"], device)
reference_state = get_reference_state(base_model)

model = base_model
if args.checkpoint:
    model, tokenizer = load_model(args.checkpoint, device)

prompts = load_imdb_prompts(tokenizer, d["n_test_prompts"],
                            d["prompt_tokens"], cfg["seed"])
wiki = load_wikipedia(d["n_wiki_samples"], cfg["seed"])
if args.max_prompts:
    prompts, wiki = prompts[:args.max_prompts], wiki[:args.max_prompts]

reward_model = RewardModel(cfg["reward"]["name"], device)
results = evaluate(model, tokenizer, reward_model, prompts, wiki,
                   reference_state, cfg, device)

os.makedirs(cfg["results_dir"], exist_ok=True)
path = f"{cfg['results_dir']}/eval_{args.name}.json"
with open(path, "w") as f:
    json.dump(results, f, indent=2)

for k, v in results.items():
    print(f"{k}: {v}")