"""Step 5: use the model. Greedy first, beam search retry if the JSON is invalid,
then add volatile fields (e.g. ts) from your pipeline instead of the model."""
import argparse, json, datetime
from transformers import T5ForConditionalGeneration
from common import load_tok, get_device, generate_batch, try_parse

_cache = {}


def load(path="ckpt/best"):
    if path not in _cache:
        dev = get_device()
        _cache[path] = (T5ForConditionalGeneration.from_pretrained(path).to(dev).eval(),
                        load_tok(path), dev)
    return _cache[path]


def to_json(instruction, path="ckpt/best", add_ts=True):
    model, tok, dev = load(path)
    for beams in (1, 4):  # fallback to beam search if greedy output is broken
        text = generate_batch(model, tok, [instruction], dev, beams=beams)[0]
        obj = try_parse(text)
        if isinstance(obj, dict):
            if add_ts:
                obj["ts"] = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
            return obj
    return {"error": "invalid_json", "raw": text}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("text")
    ap.add_argument("--model", default="ckpt/best")
    a = ap.parse_args()
    print(json.dumps(to_json(a.text, a.model), indent=2, ensure_ascii=False))
