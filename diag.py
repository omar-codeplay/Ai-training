import argparse
import json
from collections import Counter
from pathlib import Path


def iter_json_objects(text):
    dec = json.JSONDecoder()
    i, n = 0, len(text)
    while i < n:
        while i < n and text[i].isspace():
            i += 1
        if i >= n:
            break
        obj, end = dec.raw_decode(text, i)
        yield obj
        i = end


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("field")
    args = ap.parse_args()

    text = Path(args.file).read_text(encoding="utf-8")
    present = 0
    total = 0
    values = Counter()

    for obj in iter_json_objects(text):
        total += 1
        resp = obj.get("response", {})
        if args.field in resp:
            present += 1
            v = resp[args.field]
            if isinstance(v, (str, int, float, bool)) or v is None:
                values[v] += 1
            else:
                values[str(type(v).__name__)] += 1

    print(f"{args.field}: {present}/{total} rows have the key")
    for v, c in values.most_common(20):
        print(f"  {c:>6}  {v!r}")


if __name__ == "__main__":
    main()
