# Train a text -> JSON model from zero

T5-style encoder-decoder, 30M-125M params, trained from scratch on your
`{"instruction": ..., "response": {...}}` data.

## Setup
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt     # install a CUDA build of torch first if you have a GPU
```
Hardware: any GPU with 8GB+ (or Colab T4). bf16 is used when supported; fp16 is avoided on purpose (T5 overflows).

## Steps
```bash
# 1. clean + canonicalize + leak-free split  (drops ts, sorts keys, dedupes)
python prep_data.py --input dataset.jsonl --drop ts            # add --normalize if you want all keys always present

# 2. tokenizer (read the printed token counts for hex/paths/filenames)
python train_tokenizer.py --vocab 16000
#    optional extra text (HTML/CSS/JS, English) helps only if your data is small:
#    python train_tokenizer.py --extra extra_code.txt extra_english.txt

# 3. train  (tiny | small | base)
python train.py --size small --epochs 40
#    out of memory?  --bs 8 --accum 4      long outputs?  --max_tgt 1024

# 4. evaluate on held-out test
python evaluate.py --model ckpt/best
python evaluate.py --model ckpt/best --beams 4

# 5. use it
python infer.py "make a landing page with index.html and style.css, glassmorphism, #4169E1"
```

## Pick size by data size
| train rows | size | note |
|---|---|---|
| < 2k | don't train from scratch | fine-tune `t5-small`/`flan-t5-small` instead; scratch will memorize |
| 2k-20k | tiny / small | add augmentation (paraphrase instructions) |
| 20k-200k | small | |
| 200k+ | base | |

## Decisions made for you (and why)
- **Sorted keys + no spaces** in targets: one canonical output, no order jitter.
- **ts removed**: the model can't know the time; add it in code (`infer.py` does).
- **Normalize or not**: pick once. `--normalize` = every key always present. Never mix.
- **Never truncate targets**: over-long rows are dropped; a half-cut JSON teaches broken output.
- **Split by hash of instruction**: duplicate/near-duplicate prompts can't leak into test.
- **Tokenizer trained on train split only**, 16k vocab (32k wastes parameters on small data).
- **Gated-GELU FFN, tied embeddings, AdamW(0.9, 0.98), warmup+cosine, label smoothing 0.05, grad clip 1.0**.
- **Early stopping** on val NLL; best checkpoint saved separately from last.

## Debugging guide
| symptom | fix |
|---|---|
| invalid JSON | check tokenizer round-trip + max_tgt; train longer; use beams=4 |
| right keys, wrong values (hex/filenames) | more data covering them; check tokenization splits |
| train loss falls, val rises early | too little data -> more data / augmentation / smaller model / dropout 0.2 |
| loss NaN | lower lr to 1e-4; make sure not on fp16 |
| hallucinated keys | `--normalize` and add more diverse examples per task type |

## Optional upgrade: guaranteed-valid JSON
After training, wrap generation with constrained decoding (e.g. the `outlines` or
`lm-format-enforcer` libraries) using your schema, so invalid JSON becomes impossible.
