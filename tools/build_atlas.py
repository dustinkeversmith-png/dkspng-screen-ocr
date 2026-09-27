#!/usr/bin/env python3
"""
Render the glyph template atlas consumed by dks::ocr::Atlas.

    python tools/build_atlas.py [--out data/ui_fonts.dksa]

Every (font, pixel size, char) is rendered with FreeType antialiasing and stored as a raw ink
bitmap plus its placement relative to the baseline. The C++ side pushes each bitmap through the
*same* ink-extraction + feature code as live glyphs, so templates and queries can never drift.

Binary layout (little endian):
    "DKSA" u32 version=1
    u32 n_fonts   { u16 len, utf8 name }
    u32 n_glyphs  { u32 codepoint, u16 font, u16 px, i16 top (rel. baseline, +down), u16 w, u16 h,
                    u16 cap_h, u16 x_h, w*h u8 ink }
"""
from __future__ import annotations

import argparse
import struct
from pathlib import Path

from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont

from fonts import ATLAS_FONTS, CHARSET, available

SIZES = [9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 24, 30]


def cmap_of(path: Path) -> set[int]:
    f = TTFont(str(path), fontNumber=0, lazy=True)
    cps = set()
    for t in f["cmap"].tables:
        cps.update(t.cmap.keys())
    return cps


def render(font: ImageFont.FreeTypeFont, ch: str, px: int):
    pad = px
    W = px * 3 + 2 * pad
    H = px * 3 + 2 * pad
    base = pad + 2 * px
    im = Image.new("L", (W, H), 0)
    ImageDraw.Draw(im).text((pad, base), ch, fill=255, font=font, anchor="ls")
    bb = im.getbbox()
    if not bb:
        return None
    crop = im.crop(bb)
    return bb[1] - base, crop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "data" / "ui_fonts.dksa"))
    args = ap.parse_args()
    fonts = available(ATLAS_FONTS)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    names, glyphs = [], []
    for fi, path in enumerate(fonts):
        names.append(path.stem)
        cps = cmap_of(path)
        for px in SIZES:
            font = ImageFont.truetype(str(path), px)
            cap = render(font, "H", px)
            xh = render(font, "x", px)
            cap_h = cap[1].height if cap else px
            x_h = xh[1].height if xh else px // 2
            for ch in CHARSET:
                if ord(ch) not in cps:
                    continue
                r = render(font, ch, px)
                if r is None:
                    continue
                top, crop = r
                glyphs.append((ord(ch), fi, px, top, crop.width, crop.height, cap_h, x_h, crop.tobytes()))
        print(f"  {path.name:22s} {len(glyphs):6d} glyphs total")

    with open(out, "wb") as f:
        f.write(b"DKSA" + struct.pack("<I", 1))
        f.write(struct.pack("<I", len(names)))
        for n in names:
            b = n.encode("utf-8")
            f.write(struct.pack("<H", len(b)) + b)
        f.write(struct.pack("<I", len(glyphs)))
        for cp, fi, px, top, w, h, cap_h, x_h, data in glyphs:
            f.write(struct.pack("<IHHhHHHH", cp, fi, px, top, w, h, cap_h, x_h))
            f.write(data)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {len(glyphs)} glyphs, {len(names)} fonts)")


if __name__ == "__main__":
    main()
