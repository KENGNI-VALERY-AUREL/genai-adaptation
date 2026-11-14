import math

import torch

from src.model import generate
from src.utils import set_seed


@torch.no_grad()
def perplexity(model, tokenizer, texts, device, max_length=256, batch_size=8):
    """Token-level perplexity of the model on a list of texts."""
    model.eval()
    tokenizer.padding_side = "right"  # right padding for loss computation
    total_nll, total_tokens = 0.0, 0
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(texts[i:i + batch_size], return_tensors="pt",
                        padding=True, truncation=True,
                        max_length=max_length).to(device)
        logits = model(**enc).logits[:, :-1]
        labels = enc["input_ids"][:, 1:].clone()
        labels[enc["attention_mask"][:, 1:] == 0] = -100  # ignore padding
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), labels.reshape(-1),
            ignore_index=-100, reduction="sum")
        total_nll += loss.item()
        total_tokens += (labels != -100).sum().item()
    tokenizer.padding_side = "left"  # restore the setting used for generation
    return math.exp(total_nll / total_tokens)


def get_reference_state(model):
    """Copy the weights of the initial model (used for the L2 distance)."""
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters()}


def l2_distance(model, reference_state):
    """L2 distance between the current weights and the reference weights."""
    sq = 0.0
    for name, p in model.named_parameters():
        sq += (p.detach().cpu() - reference_state[name]).pow(2).sum().item()
    return math.sqrt(sq)


def distinct_n(texts, tokenizer, n):
    """Ratio of unique n-grams over all n-grams (diversity of generations)."""
    ngrams = []
    for t in texts:
        ids = tokenizer(t)["input_ids"]
        ngrams += [tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)]
    return len(set(ngrams)) / max(len(ngrams), 1)


def evaluate(model, tokenizer, reward_model, prompts, wiki_texts,
             reference_state, cfg, device):
    """Compute all metrics for one model. Same protocol for every method."""
    set_seed(cfg["seed"])  # same sampling noise for every evaluated model
    model.eval()
    e = cfg["eval"]

    # 1) Generate continuations of the fixed test prompts
    texts = []
    for _ in range(e["n_generations"]):
        for i in range(0, len(prompts), e["gen_batch_size"]):
            batch = prompts[i:i + e["gen_batch_size"]]
            texts += generate(model, tokenizer, batch,
                              max_new_tokens=cfg["model"]["max_new_tokens"],
                              device=device)

    # 2) Metrics
    rewards = reward_model(texts)
    return {
        "reward_mean": rewards.mean().item(),
        "reward_std": rewards.std().item(),
        "wiki_ppl": perplexity(model, tokenizer, wiki_texts, device,
                               max_length=e["ppl_max_length"]),
        "l2_distance": l2_distance(model, reference_state),
        "distinct_1": distinct_n(texts, tokenizer, 1),
        "distinct_2": distinct_n(texts, tokenizer, 2),
        "samples": texts[:5],
    }