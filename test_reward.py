from src.utils import set_seed, get_device, load_config
from src.reward import RewardModel

cfg = load_config()
set_seed(cfg["seed"])
device = get_device()

reward_model = RewardModel(cfg["reward"]["name"], device)

cases = {
    "very positive": "This movie was absolutely wonderful, I loved every minute of it.",
    "very negative": "This movie was terrible, a complete waste of time.",
    "neutral": "The film was released in 1998 and runs for two hours.",
    "repetitive": "great great great great great great great great",
    "off-topic": "The capital of France is Paris and it is located on the Seine.",
    "sarcasm": "Oh sure, because what I really needed was another boring sequel.",
}

rewards = reward_model(list(cases.values()))
for (name, text), r in zip(cases.items(), rewards):
    print(f"{name:15s} reward = {r:.3f} | {text[:50]}")