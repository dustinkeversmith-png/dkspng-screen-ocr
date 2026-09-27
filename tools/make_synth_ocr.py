#!/usr/bin/env python3
"""
"Rendered font grid" OCR test set: words rendered onto coloured backgrounds.

    python tools/make_synth_ocr.py [--n 600]

Writes two splits in the same manifest/gt.tsv format as the real datasets:
    datasets/synth/seen    atlas fonts, sizes *between* atlas sizes, random colours / offsets
    datasets/synth/unseen  held-out fonts the atlas never saw
Each image is one word (label "Word") plus a few multi-word lines (label "Line") for space handling.
Deterministic: fixed RNG seed.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from fonts import ATLAS_FONTS, HELDOUT_FONTS, available

WORDS = """the of and to in is you that it he was for on are as with his they at be this have from or one
had by word but not what all were we when your can said there use an each which she do how their if will up
other about out many then them these so some her would make like him into time has look two more write go see
number no way could people my than first water been call who oil its now find long down day did get come made
may part Settings File Edit View Help Window Tools Search Open Save Cancel Delete Rename Properties Options
Account Profile Sign Login Logout Password Username Email Submit Download Upload Share Print Export Import
Refresh Close Minimize Maximize Desktop Documents Pictures Music Videos Network Recent Favorites Home Back
Forward Next Previous Finish Apply Reset Default Advanced General Security Privacy Update Install Version
Windows Microsoft Chrome Firefox Google Mail Inbox Sent Drafts Trash Archive Calendar Contacts Tasks Notes
January February March April Monday Tuesday Friday Sunday Total Price Quantity Order Invoice Customer""".split()


def rand_token(rng: random.Random) -> str:
    r = rng.random()
    if r < 0.55:
        w = rng.choice(WORDS)
        m = rng.random()
        return w.upper() if m < 0.12 else (w.capitalize() if m < 0.35 else w)
    if r < 0.70:
        return str(rng.randint(0, 99999))
    if r < 0.78:
        return f"{rng.choice('$€£')}{rng.randint(1, 999)}.{rng.randint(0, 99):02d}"
    if r < 0.85:
        return f"{rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}/20{rng.randint(10, 30)}"
    if r < 0.90:
        return f"{rng.choice(WORDS).lower()}@{rng.choice(WORDS).lower()}.com"
    alpha = "ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz0123456789"
    return "".join(rng.choice(alpha) for _ in range(rng.randint(3, 8)))


def rand_colors(rng: random.Random):
    while True:
        bg = tuple(rng.randint(0, 255) for _ in range(3))
        fg = tuple(rng.randint(0, 255) for _ in range(3))
        lum = lambda c: 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]  # noqa: E731
        if abs(lum(bg) - lum(fg)) > 90:
            return bg, fg


def make_split(name: str, fonts, sizes, n: int, seed: int, root: Path):
    rng = random.Random(seed)
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    manifest = []
    for i in range(n):
        font_path = rng.choice(fonts)
        px = rng.choice(sizes)
        font = ImageFont.truetype(str(font_path), px)
        is_line = i % 5 == 4
        text = " ".join(rand_token(rng) for _ in range(rng.randint(2, 4))) if is_line else rand_token(rng)
        bb = font.getbbox(text, anchor="ls")
        w, h = bb[2] - bb[0], bb[3] - bb[1]
        mx, my = rng.randint(3, 12), rng.randint(3, 10)
        W, H = w + 2 * mx, h + 2 * my
        bg, fg = rand_colors(rng)
        im = Image.new("RGB", (W, H), bg)
        ox = mx - bb[0] + rng.random()  # sub-pixel phase
        oy = my - bb[1]
        ImageDraw.Draw(im).text((ox, oy), text, fill=fg, font=font, anchor="ls")
        stem = f"{i:05d}"
        im.save(d / f"{stem}.png")
        with open(d / f"{stem}.gt.tsv", "w", encoding="utf-8", newline="\n") as f:
            f.write(f"{mx}\t{my}\t{w}\t{h}\t{'Line' if is_line else 'Word'}\t{text}\t{font_path.stem}\t{px}\n")
        manifest.append(f"{stem}.png\t{stem}.gt.tsv\n")
    (d / "manifest.tsv").write_text("".join(manifest), encoding="utf-8", newline="\n")
    print(f"  {name}: {n} samples, {len(fonts)} fonts")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600)
    args = ap.parse_args()
    root = Path(__file__).resolve().parent.parent / "datasets" / "synth"
    # Atlas sizes are 9..16,18,20,24,30 -> test on sizes not in the atlas plus a few inside.
    make_split("seen", available(ATLAS_FONTS), [11, 13, 17, 19, 22, 26, 28], args.n, 1, root)
    make_split("unseen", available(HELDOUT_FONTS), [11, 13, 15, 17, 19, 22, 26], args.n, 2, root)


if __name__ == "__main__":
    main()
