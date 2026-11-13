import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class RewardModel:
    """Sentiment classifier used as a reward model: reward = P(positive) in [0, 1]."""

    def __init__(self, name: str, device: str = "cpu"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name)
        self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def score(self, texts, batch_size: int = 32):
        """Return the raw score s = logit_pos - logit_neg for each text."""
        scores = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self.tokenizer(
                batch, return_tensors="pt", padding=True,
                truncation=True, max_length=256,
            ).to(self.device)
            logits = self.model(**inputs).logits  # shape: [batch, 2] = (neg, pos)
            scores.append((logits[:, 1] - logits[:, 0]).cpu())
        return torch.cat(scores)

    def __call__(self, texts, batch_size: int = 32):
        """Return the reward r = sigmoid(s) = P(positive) for each text."""
        return torch.sigmoid(self.score(texts, batch_size))