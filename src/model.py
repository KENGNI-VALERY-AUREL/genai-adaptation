import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(name: str = "gpt2", device: str = "cpu"):
    """Load GPT-2 (124M) and its tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(name)
    # GPT-2 has no padding token: reuse the end-of-sequence token
    tokenizer.pad_token = tokenizer.eos_token
    # Left padding is required for correct batched generation
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(name)
    model.to(device)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def generate(model, tokenizer, prompts, max_new_tokens=20,
             temperature=1.0, top_k=50, device="cpu"):
    
    """Generate a continuation for each prompt. Return the list of full texts."""
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,  # sampling is needed later for GRPO (diverse responses)
        temperature=temperature,
        top_k=top_k,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)