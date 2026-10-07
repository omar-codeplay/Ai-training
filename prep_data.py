#!/usr/bin/env python3
"""Clean + canonicalize dataset.jsonl -> dataset.fixed.jsonl."""
import argparse, json, re, sys
from pathlib import Path

STYLE_COLORS = {
    "dark minimal": ["#18181b", "#4169E1"],
    "minimal": ["#ffffff", "#8A2BE2"],
    "modern minimal": ["#ffffff", "#4c8dff"],
    "clean corporate": ["#f5f5f5", "#4c8dff"],
    "clean": ["#ffffff", "#2563eb"],
    "corporate": ["#f8fafc", "#2563eb"],
    "brutalist": ["#000000", "#FF0000"],
    "brutalist bold": ["#000000", "#FF0000"],
    "neo-brutalist": ["#FFE600", "#000000"],
    "glassmorphism": ["#ffffff", "#00BFFF"],
    "dark theme glassy": ["#0d1117", "#00BFFF"],
    "glass": ["#ffffff", "#7C3AED"],
    "neumorphic soft": ["#f3f4f6", "#7C3AED"],
    "neumorphism": ["#e0e5ec", "#6d5dfc"],
    "flat material": ["#ffffff", "#32CD32"],
    "flat": ["#3498DB", "#E74C3C"],
    "material": ["#6200EE", "#03DAC6"],
    "material you": ["#6750A4", "#EADDFF"],
    "playful colorful": ["#FFD1DC", "#FFECB3", "#B2DFDB"],
    "playful": ["#FF6B6B", "#4ECDC4"],
    "retro neon": ["#FF00FF", "#00FFFF"],
    "retro": ["#F7B267", "#F4845F"],
    "synthwave": ["#FF00A0", "#00F0FF"],
    "pastel soft": ["#FFD1DC", "#BBDEFB"],
    "pastel": ["#FADADD", "#B5EAD7"],
    "dark": ["#0b0b0b", "#22d3ee"],
    "midnight": ["#0f172a", "#38bdf8"],
    "cyberpunk": ["#0b0b0b", "#f6ff00"],
    "techy mono": ["#1a1a1a", "#00FF7F"],
    "mono": ["#111111", "#eeeeee"],
    "elegant serif": ["#fafafa", "#9370DB"],
    "elegant": ["#f7f5f0", "#8b6f47"],
    "luxury": ["#0e0e0e", "#c6a664"],
    "vintage": ["#F5E6C8", "#A0522D"],
    "gradient modern": ["#FFB347", "#4169E1"],
    "gradient": ["#a855f7", "#ec4899"],
    "aurora": ["#00c6ff", "#0072ff"],
    "sunset": ["#ff9966", "#ff5e62"],
    "ocean": ["#2E3192", "#1BFFFF"],
    "forest": ["#11998e", "#38ef7d"],
    "high-contrast accessible": ["#000000", "#FFD700"],
    "accessible": ["#0b0b0b", "#ffd400"],
    "skeuomorphic": ["#e6d7b8", "#8b5a2b"],
    "paper": ["#F5F0E6", "#5C4033"],
    "saas": ["#0f172a", "#3b82f6"],
    "blog": ["#fafafa", "#9370DB"],
    "docs": ["#f8fafc", "#2563eb"],
    "portfolio": ["#0a0a0a", "#f59e0b"],
    "web server": ["#0b0f19", "#22d3ee"],
}
DEFAULT_COLORS = ["#ffffff", "#4c8dff"]

STYLE_SYNONYMS = {
    "glassy": "dark theme glassy",
    "dark glassy": "dark theme glassy",
    "glassmorphic": "glassmorphism",
    "frosted glass": "glassmorphism",
    "brutalism": "brutalist bold",
    "soft neumorphic": "neumorphic soft",
    "material design": "material",
    "md3": "material you",
    "flat ui": "flat",
    "playful": "playful colorful",
    "colorful": "playful colorful",
    "pastel": "pastel soft",
    "serif": "elegant serif",
    "neon": "retro neon",
    "synth": "synthwave",
    "terminal": "techy mono",
    "clean corp": "clean corporate",
    "modern clean": "modern minimal",
    "minimal modern": "modern minimal",
    "a11y": "high-contrast accessible",
}
DROP_KEYS_DEFAULT = ("ts", "precision", "website")
SEP = " -- "

TAIL_RE = re.compile(
    r"[\s,;-]*(?:colors:\s*\([^)]*\))?[\s,;-]*(?:design_type:\s*[\w\- ]+)?"
    r"[\s.,;-]*$",
    re.IGNORECASE,
)
LINE_AT_RE = re.compile(r"\s*,?\s*at\s+line\s+\d+", re.IGNORECASE)
LINE_ONLY_RE = re.compile(r"\s*,?\s*line\s+\d+", re.IGNORECASE)
ERR_RE = re.compile(r"\s*-+\s*[A-Z][\w.]*(?:Error|Exception)[^\n,]*", re.IGNORECASE)


def iter_json_objects(text):
    for lineno, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            fixed = re.sub(r",(\s*[}\]])", r"\1", line)
            try:
                yield json.loads(fixed)
                print("  fixed trailing comma on line %d" % lineno)
            except json.JSONDecodeError as e:
                print("  SKIP line %d: %s" % (lineno, e.msg))
                continue


def resolve_style(resp, instruction=""):
    candidates = []
    s = resp.get("style")
    if isinstance(s, str):
        candidates.append(s)
    dev = resp.get("device", "")
    if isinstance(dev, str):
        m = re.search(r"style:([^,]+)", dev)
        if m:
            candidates.append(m.group(1))
    for c in candidates:
        low = c.strip().lower()
        if low in STYLE_COLORS:
            return low
        if low in STYLE_SYNONYMS:
            return STYLE_SYNONYMS[low]
        for k in STYLE_COLORS:
            if k in low:
                return k
    text = (instruction or "").lower()
    for syn, target in STYLE_SYNONYMS.items():
        if syn in text:
            return target
    for k in STYLE_COLORS:
        if k in text:
            return k
    return None


def style_colors(style):
    if not style:
        return DEFAULT_COLORS
    s = style.lower()
    if s in STYLE_COLORS:
        return STYLE_COLORS[s]
    if s in STYLE_SYNONYMS:
        return STYLE_COLORS.get(STYLE_SYNONYMS[s], DEFAULT_COLORS)
    return DEFAULT_COLORS


def strip_website_tail(text):
    text = text.replace("\u2014", " -- ").replace("\u2013", " -- ")
    return TAIL_RE.sub("", text).rstrip(" ,;.-")


def strip_debug_tail(text):
    text = text.replace("\u2014", " -- ").replace("\u2013", " -- ")
    text = ERR_RE.sub("", text)
    text = LINE_AT_RE.sub("", text)
    text = LINE_ONLY_RE.sub("", text)
    return text.rstrip(" ,;.-")


def fix_website(obj, resp):
    design = resp.get("design_type") or resolve_style(resp, obj.get("instruction", ""))
    if design:
        resp["design_type"] = design
    colors = resp.get("colors") or style_colors(design)
    resp["colors"] = colors
    head = strip_website_tail(obj["instruction"])
    parts = []
    if len(colors) >= 2:
        parts.append("colors: (%s, %s)" % (colors[0], colors[1]))
    if design:
        parts.append("design_type: %s" % design)
    obj["instruction"] = head + SEP + ", ".join(parts) if parts else head


def fix_debug(obj, resp):
    name = resp.get("file_name") or resp.get("name") or "main.py"
    line = resp.get("line") or 0
    err = resp.get("error") or ""
    head = strip_debug_tail(obj["instruction"])
    if line and " at line " not in head.lower():
        if name in head:
            head = head.replace(name, name + " at line " + str(line), 1)
        else:
            head = head + " at line " + str(line)
    if err and err not in head:
        head = head + SEP + err
    obj["instruction"] = head


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="dataset.jsonl")
    ap.add_argument("--output", default="dataset.fixed.jsonl")
    ap.add_argument("--drop", nargs="*", default=list(DROP_KEYS_DEFAULT))
    args = ap.parse_args()
    src = Path(args.input)
    dst = Path(args.output)
    if not src.exists():
        print("error: %s not found" % src)
        sys.exit(1)
    drop = [k for k in args.drop if k != "colors"]
    stats = {"website": 0, "debug": 0, "other": 0, "kept": 0}
    with dst.open("w", encoding="utf-8") as out:
        for obj in iter_json_objects(src.read_text(encoding="utf-8")):
            resp = obj.get("response", {})
            for k in drop:
                resp.pop(k, None)
            if resp.get("ai") == "website":
                fix_website(obj, resp)
                stats["website"] += 1
            elif resp.get("action") == "debug":
                fix_debug(obj, resp)
                stats["debug"] += 1
            else:
                stats["other"] += 1
            out.write(json.dumps(obj, ensure_ascii=False) + "\n")
            stats["kept"] += 1
    print("wrote %d rows -> %s" % (stats["kept"], dst))
    print("  website rows: %d" % stats["website"])
    print("  debug rows:   %d" % stats["debug"])
    print("  other rows:   %d" % stats["other"])


if __name__ == "__main__":
    main()
