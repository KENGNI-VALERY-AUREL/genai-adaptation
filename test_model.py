from src.utils import set_seed, get_device, load_config
from src.model import load_model, generate

cfg = load_config()
set_seed(cfg["seed"])
device = get_device()

model, tokenizer = load_model(cfg["model"]["name"], device)
n_params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"Parameters: {n_params:.1f}M")

prompts = ["I am what", "I went to see this film and"]
outputs = generate(model, tokenizer, prompts,
                   max_new_tokens=cfg["model"]["max_new_tokens"], device=device)
for o in outputs:
    print("-", o)