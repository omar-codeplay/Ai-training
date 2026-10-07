"""Step 4: evaluate on the held-out TEST split (run once, after you've finished tuning)."""
import argparse, json, collections
from transformers import T5ForConditionalGeneration
from common import read_jsonl, load_tok, get_device, generate_batch, try_parse, canon


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ckpt/best")
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--split", default="test")
    ap.add_argument("--beams", type=int, default=1)
    ap.add_argument("--max_new", type=int, default=512)
    ap.add_argument("--show_fail", type=int, default=5)
    args = ap.parse_args()

    device = get_device()
    tok = load_tok(args.model)
    model = T5ForConditionalGeneration.from_pretrained(args.model).to(device).eval()
    rows = read_jsonl(f"{args.data_dir}/{args.split}.jsonl")
    schema_keys = set(json.load(open(f"{args.data_dir}/schema.json"))["keys"])

    outs = generate_batch(model, tok, [r["src"] for r in rows], device,
                          beams=args.beams, max_new=args.max_new)
    n = len(rows)
    valid = exact = keyset = no_halluc = 0
    key_ok, key_tot = collections.Counter(), collections.Counter()
    fails = []
    for r, o in zip(rows, outs):
        gold = json.loads(r["tgt"])
        pred = try_parse(o)
        if not isinstance(pred, dict):
            fails.append((r["src"], r["tgt"], o)); continue
        valid += 1
        exact += canon(pred) == r["tgt"]
        keyset += set(pred) == set(gold)
        no_halluc += set(pred) <= schema_keys
        for k, v in gold.items():
            key_tot[k] += 1
            key_ok[k] += pred.get(k) == v
        if canon(pred) != r["tgt"] and len(fails) < args.show_fail:
            fails.append((r["src"], r["tgt"], o))

    print(f"\n{args.split} rows: {n}  beams: {args.beams}")
    print(f"valid JSON      {valid/n:7.2%}   (target 100%)")
    print(f"exact match     {exact/n:7.2%}   (target >=95%)")
    print(f"key-set match   {keyset/n:7.2%}   (target 100%)")
    print(f"no hallucinated {no_halluc/n:7.2%}   (target 100%)")
    print("\nper-key value accuracy (worst first):")
    for k, tot in sorted(key_tot.items(), key=lambda kv: key_ok[kv[0]] / kv[1]):
        print(f"  {k:<20} {key_ok[k]/tot:7.2%}  ({tot})")
    if fails:
        print("\nexamples of failures:")
        for s, g, p in fails[:args.show_fail]:
            print(f"- SRC : {s[:200]}\n  GOLD: {g[:300]}\n  PRED: {p[:300]}\n")


if __name__ == "__main__":
    main()
