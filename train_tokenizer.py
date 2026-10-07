"""Step 2: train a byte-level BPE tokenizer and sanity-check it.

Corpus = train src + train tgt (+ any --extra text files: HTML/CSS/JS, English, hex, paths...).
Only the TRAIN split is used, so nothing leaks from val/test.
"""
import argparse, os, json
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from transformers import PreTrainedTokenizerFast
from common import read_jsonl, SPECIALS

CHECKS = ["#4169E1", "index.html", "style.css", "glassmorphism",
          "html/css in separate files", "2024-12-05T17:16:28Z",
          '{"file_name":"index.html"}']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--extra", nargs="*", default=[], help="extra plain-text files")
    ap.add_argument("--vocab", type=int, default=16000,
                    help="16k is plenty for small data; 32k only with lots of extra text")
    ap.add_argument("--out", default="tok")
    args = ap.parse_args()

    rows = read_jsonl(f"{args.data_dir}/train.jsonl")
    os.makedirs("tmp", exist_ok=True)
    with open("tmp/corpus.txt", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(r["src"].replace("\n", " \n ") + "\n")
            f.write(r["tgt"] + "\n")

    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab, special_tokens=SPECIALS, min_frequency=2,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=True)
    tok.train(["tmp/corpus.txt"] + args.extra, trainer)

    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok, bos_token="<bos>", eos_token="<eos>",
        pad_token="<pad>", unk_token="<unk>")
    fast.save_pretrained(args.out)

    assert [fast.convert_tokens_to_ids(s) for s in SPECIALS] == [0, 1, 2, 3]
    print("\nTokenization check (fewer tokens = better):")
    for s in CHECKS:
        t = fast.tokenize(s)
        print(f"  {len(t):>2} tokens  {s!r} -> {t}")
    # round trip + length stats on real targets
    lens = []
    for r in rows[:2000]:
        ids = fast(r["tgt"])["input_ids"]
        assert fast.decode(ids) == r["tgt"], "round-trip failed!"
        lens.append(len(ids))
    lens.sort()
    print(f"\ntarget token length: median {lens[len(lens)//2]}, "
          f"p95 {lens[int(len(lens)*.95)]}, max {lens[-1]}  (set --max_tgt above p99)")


if __name__ == "__main__":
    main()
