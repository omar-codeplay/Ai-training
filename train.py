"""Step 3: train a T5-style encoder-decoder from scratch.

Implements what the spec only described: real LR warmup + cosine decay, label smoothing,
gradient accumulation, bf16, best-checkpoint saving, early stopping, and periodic
exact-match checks on generated JSON (val loss alone doesn't tell you JSON quality).
"""
import argparse, json, math, os, random, time
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from transformers import T5Config, T5ForConditionalGeneration, get_cosine_schedule_with_warmup
from common import (read_jsonl, load_tok, get_device, autocast_ctx,
                    generate_batch, try_parse, canon)

PRESETS = {  # approx params with a 16k vocab
    "tiny":  dict(d_model=384, d_ff=1536, num_layers=4, num_decoder_layers=4, num_heads=6),   # ~30M
    "small": dict(d_model=512, d_ff=2048, num_layers=6, num_decoder_layers=6, num_heads=8),   # ~55M
    "base":  dict(d_model=768, d_ff=3072, num_layers=6, num_decoder_layers=6, num_heads=12),  # ~125M
}


class PairDS(Dataset):
    def __init__(self, rows, tok, max_src, max_tgt):
        self.items, dropped = [], 0
        for r in rows:
            s = tok(r["src"], add_special_tokens=False)["input_ids"]
            t = tok(r["tgt"], add_special_tokens=False)["input_ids"]
            if len(s) > max_src - 1 or len(t) > max_tgt - 1:
                dropped += 1  # never truncate JSON: a cut target teaches broken output
                continue
            self.items.append((s + [tok.eos_token_id], t + [tok.eos_token_id]))
        if dropped:
            print(f"  dropped {dropped} rows longer than max_src/max_tgt")

    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]


def make_collate(pad_id):
    def collate(batch):
        src = pad_sequence([torch.tensor(b[0]) for b in batch], batch_first=True, padding_value=pad_id)
        tgt = pad_sequence([torch.tensor(b[1]) for b in batch], batch_first=True, padding_value=-100)
        return {"input_ids": src, "attention_mask": (src != pad_id).long(), "labels": tgt}
    return collate


def loss_fn(model, batch, smoothing):
    out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                labels=batch["labels"])
    logits = out.logits.float()
    return F.cross_entropy(logits.view(-1, logits.size(-1)), batch["labels"].view(-1),
                           ignore_index=-100, label_smoothing=smoothing)


@torch.no_grad()
def val_loss(model, dl, device):
    model.eval()
    tot, n = 0.0, 0
    for b in dl:
        b = {k: v.to(device) for k, v in b.items()}
        with autocast_ctx(device):
            tot += loss_fn(model, b, 0.0).item()  # no smoothing for eval -> true NLL
        n += 1
    model.train()
    return tot / max(n, 1)


def quick_exact(model, tok, rows, device, n=200, max_new=512):
    sub = rows[:n]
    outs = generate_batch(model, tok, [r["src"] for r in sub], device, beams=1, max_new=max_new)
    valid = exact = 0
    for r, o in zip(sub, outs):
        p = try_parse(o)
        if p is not None:
            valid += 1
            exact += canon(p) == r["tgt"]
    model.train()
    return valid / len(sub), exact / len(sub)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--tok_dir", default="tok")
    ap.add_argument("--out", default="ckpt")
    ap.add_argument("--size", default="small", choices=PRESETS)
    ap.add_argument("--max_src", type=int, default=256)
    ap.add_argument("--max_tgt", type=int, default=512)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--accum", type=int, default=2, help="effective batch = bs*accum")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--smoothing", type=float, default=0.05)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=6, help="early-stop on val loss")
    ap.add_argument("--eval_every", type=int, default=2, help="epochs between exact-match checks")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", default=None, help="path to a saved model dir")
    args = ap.parse_args()

    random.seed(args.seed); torch.manual_seed(args.seed)
    device = get_device()
    tok = load_tok(args.tok_dir)
    print("device:", device, "| vocab:", len(tok))

    train_rows = read_jsonl(f"{args.data_dir}/train.jsonl")
    val_rows = read_jsonl(f"{args.data_dir}/val.jsonl")
    print("building datasets...")
    tr = PairDS(train_rows, tok, args.max_src, args.max_tgt)
    va = PairDS(val_rows, tok, args.max_src, args.max_tgt)
    col = make_collate(tok.pad_token_id)
    train_dl = DataLoader(tr, batch_size=args.bs, shuffle=True, collate_fn=col, drop_last=True)
    val_dl = DataLoader(va, batch_size=args.bs, shuffle=False, collate_fn=col)

    if args.resume:
        model = T5ForConditionalGeneration.from_pretrained(args.resume)
    else:
        cfg = T5Config(
            vocab_size=len(tok), d_kv=64, dropout_rate=args.dropout,
            decoder_start_token_id=tok.bos_token_id, pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id, tie_word_embeddings=True,
            feed_forward_proj="gated-gelu",  # T5 v1.1 style: better than ReLU FFN
            **PRESETS[args.size])
        model = T5ForConditionalGeneration(cfg)
    model.to(device).train()
    print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    decay = [p for n, p in model.named_parameters() if p.ndim >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.98), eps=1e-8)
    steps_per_epoch = max(len(train_dl) // args.accum, 1)
    total_steps = steps_per_epoch * args.epochs
    sched = get_cosine_schedule_with_warmup(opt, min(args.warmup, total_steps // 10), total_steps)
    print(f"optimizer steps/epoch {steps_per_epoch}, total {total_steps}")

    os.makedirs(args.out, exist_ok=True)
    best, bad, step = float("inf"), 0, 0
    log = []
    for epoch in range(args.epochs):
        t0, run, cnt = time.time(), 0.0, 0
        for i, batch in enumerate(train_dl):
            batch = {k: v.to(device) for k, v in batch.items()}
            with autocast_ctx(device):
                loss = loss_fn(model, batch, args.smoothing) / args.accum
            loss.backward()
            run += loss.item() * args.accum; cnt += 1
            if (i + 1) % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
        vl = val_loss(model, val_dl, device)
        msg = (f"epoch {epoch+1:>3}/{args.epochs} train {run/cnt:.4f} val_nll {vl:.4f} "
               f"lr {sched.get_last_lr()[0]:.2e} {time.time()-t0:.0f}s")
        if (epoch + 1) % args.eval_every == 0 and val_rows:
            v, e = quick_exact(model, tok, val_rows, device, max_new=args.max_tgt)
            msg += f" | val valid-JSON {v:.1%} exact {e:.1%}"
        print(msg, flush=True)
        log.append({"epoch": epoch + 1, "train": run / cnt, "val": vl})

        if vl < best - 1e-4:
            best, bad = vl, 0
            model.save_pretrained(f"{args.out}/best")
            tok.save_pretrained(f"{args.out}/best")
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stop (no val improvement for {args.patience} epochs)")
                break
    model.save_pretrained(f"{args.out}/last")
    json.dump(log, open(f"{args.out}/log.json", "w"), indent=1)
    print(f"done. best val_nll {best:.4f} -> {args.out}/best")


if __name__ == "__main__":
    main()
