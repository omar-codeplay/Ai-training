"""Shared helpers for the text->JSON seq2seq project."""
import json
import torch
from transformers import PreTrainedTokenizerFast

SPECIALS = ["<pad>", "<unk>", "<bos>", "<eos>"]  # ids 0,1,2,3 (order matters)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_tok(path="tok"):
    return PreTrainedTokenizerFast.from_pretrained(path)


def canon(obj):
    """Canonical JSON string: sorted keys, no spaces. Used for targets AND comparisons."""
    return json.dumps(obj, separators=(",", ":"), sort_keys=True, ensure_ascii=False)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def autocast_ctx(device):
    """bf16 on supported GPUs, otherwise fp32. (T5 is unstable in fp16 - never use fp16.)"""
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return torch.autocast(device.type, enabled=False)


@torch.no_grad()
def generate_batch(model, tok, srcs, device, beams=1, max_new=512, max_src=256, bs=32):
    """Generate decoded strings for a list of source strings."""
    model.eval()
    outs = []
    order = sorted(range(len(srcs)), key=lambda i: len(srcs[i]))  # less padding
    results = [None] * len(srcs)
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        enc = tok([srcs[i] for i in idx], return_tensors="pt", padding=True,
                  truncation=True, max_length=max_src - 1)
        # append <eos> to each source (T5 convention), matching training
        ids, mask = enc["input_ids"], enc["attention_mask"]
        eos = torch.full((ids.size(0), 1), tok.eos_token_id)
        pad = torch.full((ids.size(0), 1), tok.pad_token_id)
        lens = mask.sum(1)
        ids = torch.cat([ids, pad], 1)
        mask = torch.cat([mask, torch.zeros_like(pad)], 1)
        for r in range(ids.size(0)):
            ids[r, lens[r]] = tok.eos_token_id
            mask[r, lens[r]] = 1
        with autocast_ctx(device):
            gen = model.generate(
                input_ids=ids.to(device), attention_mask=mask.to(device),
                max_new_tokens=max_new, num_beams=beams, do_sample=False,
                eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id,
            )
        texts = tok.batch_decode(gen, skip_special_tokens=True)
        for i, t in zip(idx, texts):
            results[i] = t
    return results


def try_parse(text):
    try:
        return json.loads(text)
    except Exception:
        return None
