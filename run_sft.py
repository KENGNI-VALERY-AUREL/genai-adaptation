import argparse

from src.utils import set_seed, get_device, load_config
from src.model import load_model
from src.data import load_imdb_positive
from src.sft import train_sft

parser = argparse.ArgumentParser()
parser.add_argument("--max-samples", type=int, default=None,
                    help="use only N training reviews (quick test)")
args = parser.parse_args()

cfg = load_config()
set_seed(cfg["seed"])
device = get_device()

model, tokenizer = load_model(cfg["model"]["name"], device)
d = cfg["data"]
train, val = load_imdb_positive(d["n_train"], d["n_val"], cfg["seed"])

if args.max_samples:
    train, val = train[:args.max_samples], val[:args.max_samples // 4]

stats = train_sft(model, tokenizer, train, val, cfg, device)
print(stats)