"""Step 1: clean + canonicalize data and make leak-free splits.

Input  : jsonl with {"instruction": str, "response": dict (or JSON string)}
Output : data/train.jsonl, val.jsonl, test.jsonl with {"src","tgt"} + data/schema.json

What it fixes for you:
  * sorts keys everywhere (no field-order jitter)
  * drops volatile keys (default: ts) - add them back in your pipeline
  * optional --normalize: every row gets every top-level key (defaults filled)
  * dedupes identical instructions; drops ones with conflicting outputs
  * splits by hash of the instruction so near-duplicates never leak into test
"""
import argparse, hashlib, json, os, collections
from common import read_jsonl, write_jsonl, canon


def drop_keys(o, keys):
    if isinstance(o, dict):
        return {k: drop_keys(v, keys) for k, v in o.items() if k not in keys}
    if isinstance(o, list):
        return [drop_keys(v, keys) for v in o]
    return o


def default_for(v):
    if isinstance(v, bool): return False
    if isinstance(v, (int, float)): return 0
    if isinstance(v, list): return []
    if isinstance(v, dict): return {}
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out_dir", default="data")
    ap.add_argument("--drop", nargs="*", default=["ts"])
    ap.add_argument("--normalize", action="store_true",
                    help="emit ALL top-level keys in every row (defaults for missing)")
    args = ap.parse_args()

    rows = read_jsonl(args.input)
    cleaned, bad = [], 0
    for r in rows:
        try:
            src = r["instruction"].strip()
            resp = r["response"]
            if isinstance(resp, str):
                resp = json.loads(resp)
            resp = drop_keys(resp, set(args.drop))
            cleaned.append((src, resp))
        except Exception:
            bad += 1
    print(f"rows: {len(rows)}  unparseable/skipped: {bad}")

    # optional schema normalization
    key_types = collections.defaultdict(collections.Counter)
    for _, resp in cleaned:
        if isinstance(resp, dict):
            for k, v in resp.items():
                key_types[k][type(v).__name__] += 1
    key_defaults = {}
    for k, c in key_types.items():
        sample = next(r[k] for _, r in cleaned if isinstance(r, dict) and k in r
                      and type(r[k]).__name__ == c.most_common(1)[0][0])
        key_defaults[k] = default_for(sample)
    if args.normalize:
        cleaned = [(s, {**{k: d for k, d in key_defaults.items()}, **r}) for s, r in cleaned]

    # dedupe by instruction, drop conflicts
    by_src = collections.defaultdict(set)
    for s, r in cleaned:
        by_src[s].add(canon(r))
    pairs, conflicts = [], 0
    for s, tg in by_src.items():
        if len(tg) == 1:
            pairs.append({"src": s, "tgt": next(iter(tg))})
        else:
            conflicts += 1
    print(f"unique instructions: {len(by_src)}  conflicting (dropped): {conflicts}")

    # hash split
    train, val, test = [], [], []
    for p in pairs:
        h = int(hashlib.md5(p["src"].encode()).hexdigest(), 16) % 100
        (train if h < 90 else val if h < 95 else test).append(p)
    os.makedirs(args.out_dir, exist_ok=True)
    write_jsonl(f"{args.out_dir}/train.jsonl", train)
    write_jsonl(f"{args.out_dir}/val.jsonl", val)
    write_jsonl(f"{args.out_dir}/test.jsonl", test)
    with open(f"{args.out_dir}/schema.json", "w") as f:
        json.dump({"keys": sorted(key_defaults), "defaults": key_defaults,
                   "normalized": args.normalize}, f, indent=2)
    print(f"train {len(train)}  val {len(val)}  test {len(test)}")
    print("key set:", sorted(key_defaults))
    if len(train) < 2000:
        print("WARNING: <2k training rows. From-scratch training will likely memorize;"
              " consider more data/augmentation or fine-tuning t5-small instead.")


if __name__ == "__main__":
    main()
