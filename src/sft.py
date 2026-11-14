import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.logger import CSVLogger


def make_loader(texts, tokenizer, max_length, batch_size, shuffle):
    """Build a DataLoader that tokenizes a batch of raw texts on the fly."""

    def collate(batch):
        enc = tokenizer(batch, return_tensors="pt", padding=True,
                        truncation=True, max_length=max_length)
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100  # ignore padding in the loss
        enc["labels"] = labels
        return enc

    return DataLoader(texts, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=collate)


@torch.no_grad()
def eval_loss(model, loader, device):
    """Mean language-modeling loss over a loader."""
    model.eval()
    losses = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        losses.append(model(**batch).loss.item())
    return sum(losses) / len(losses)


def train_sft(model, tokenizer, train_texts, val_texts, cfg, device,
              results_dir="results", run_name="sft"):
    """Supervised fine-tuning on positive reviews. Saves a checkpoint and a CSV log."""
    sft_cfg = cfg["sft"]
    # Right padding is required for training (left padding is only for generation)
    tokenizer.padding_side = "right"

    train_loader = make_loader(train_texts, tokenizer, sft_cfg["max_length"],
                               sft_cfg["batch_size"], shuffle=True)
    val_loader = make_loader(val_texts, tokenizer, sft_cfg["max_length"],
                             sft_cfg["batch_size"], shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(sft_cfg["lr"]))
    logger = CSVLogger(f"{results_dir}/{run_name}_log.csv",
                       ["step", "epoch", "train_loss", "val_loss"])

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = time.time()
    step = 0

    for epoch in range(sft_cfg["epochs"]):
        model.train()
        for batch in tqdm(train_loader, desc=f"epoch {epoch + 1}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            step += 1
            logger.log({"step": step, "epoch": epoch + 1,
                        "train_loss": round(loss.item(), 4), "val_loss": ""})

        val = eval_loss(model, val_loader, device)
        logger.log({"step": step, "epoch": epoch + 1,
                    "train_loss": "", "val_loss": round(val, 4)})
        print(f"epoch {epoch + 1}: val_loss = {val:.4f}")

    train_time = time.time() - start
    peak_mem = (torch.cuda.max_memory_allocated() / 1e9
                if device == "cuda" else None)

    model.save_pretrained(sft_cfg["checkpoint_dir"])
    tokenizer.save_pretrained(sft_cfg["checkpoint_dir"])
    tokenizer.padding_side = "left"  # restore the setting used for generation

    print(f"Training time: {train_time / 60:.1f} min | "
          f"peak GPU memory: {peak_mem if peak_mem is None else round(peak_mem, 2)} GB")
    return {"train_time_s": train_time, "peak_gpu_mem_gb": peak_mem}