#!/usr/bin/env python3
"""
Render the math symbol atlas (data/math_fonts.dksa) from the math fonts in fonts/.

    python tools/build_math_atlas.py [--fonts fonts] [--out data/math_fonts.dksa]

Same binary format as the UI atlas (tools/build_atlas.py), so dks::ocr::Atlas / Classifier load it
unchanged. The stored codepoint is the *label*, not necessarily the rendered codepoint: math-italic
U+1D465 (𝑥) is stored as 'x', math-italic Greek U+1D6FC (𝛼) as U+03B1 (α), calligraphic 𝒜 as 'A'.
LaTeX names for labels live in include/dks/latex/math_symbols.hpp.
"""
from __future__ import annotations

import argparse
import struct
from pathlib import Path

from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
SIZES = [8, 9, 10, 11, 12, 13, 14, 16, 18, 20, 24, 30, 40]


def math_charset():
    """[(label_codepoint, [render_codepoints...])]"""
    out = {}

    def add(label, *renders):
        out.setdefault(ord(label) if isinstance(label, str) else label, []).extend(
            ord(r) if isinstance(r, str) else r for r in renders)

    # Latin: math italic (default in math mode), upright (\mathrm, dx), bold (\bf), calligraphic (\cal).
    for i in range(26):
        lo, up = chr(ord('a') + i), chr(ord('A') + i)
        add(lo, 0x210E if lo == 'h' else 0x1D44E + i, lo, 0x1D41A + i)
        add(up, 0x1D434 + i, up, 0x1D400 + i)
        cal = {'B': 0x212C, 'E': 0x2130, 'F': 0x2131, 'H': 0x210B, 'I': 0x2110, 'L': 0x2112, 'M': 0x2133, 'R': 0x211B}
        add(up, cal.get(up, 0x1D49C + i))
    for d in "0123456789":
        add(d, d)
    # Greek lower case: math italic U+1D6FC.. <-> U+03B1..U+03C9, plus the variant forms.
    for i in range(25):
        add(0x03B1 + i, 0x1D6FC + i)
    for label, render in ((0x03F5, 0x1D716), (0x03D1, 0x1D717), (0x03D5, 0x1D719), (0x03F1, 0x1D71A), (0x03D6, 0x1D71B)):
        add(label, render)
    # Greek upper case is upright in LaTeX.
    for c in "ΓΔΘΛΞΠΣΥΦΨΩ":
        add(c, c)
    ops = ("+-=<>()[]{}|/,.;:!'*&#\"~"
           "±∓×÷·⋅∗∘•⋆≤≥≠≈≡∼≃≅∝≪≫∞∂∇∑∏∫∮√→←↔⇒⇔⟶↦↑↓∈∉⊂⊃⊆⊇∪∩∀∃¬∧∨⊗⊕⊙†‡‖⟨⟩ℏℓ…⋯⋮′⊥∥△ℑℜ℘ℵ♯♭⊔⊓⋄⇀")
    for c in ops:
        add(c, c)
    add('-', 0x2212)  # minus sign renders differently from the hyphen
    return sorted(out.items())


def cmap_of(path: Path) -> set[int]:
    return set(TTFont(str(path), fontNumber=0, lazy=True).getBestCmap().keys())


def render(font, cp, px):
    pad = px
    W = H = px * 4 + 2 * pad
    base = pad + 3 * px
    im = Image.new("L", (W, H), 0)
    ImageDraw.Draw(im).text((pad, base), chr(cp), fill=255, font=font, anchor="ls")
    bb = im.getbbox()
    if not bb:
        return None
    return bb[1] - base, im.crop(bb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fonts", default=str(ROOT / "fonts"))
    ap.add_argument("--out", default=str(ROOT / "data" / "math_fonts.dksa"))
    args = ap.parse_args()
    fonts = sorted(p for p in Path(args.fonts).iterdir() if p.suffix.lower() in (".otf", ".ttf"))
    if not fonts:
        raise SystemExit(f"no .otf/.ttf fonts in {args.fonts}")
    charset = math_charset()
    names, glyphs = [], []
    for fi, path in enumerate(fonts):
        names.append(path.stem)
        cps = cmap_of(path)
        n0 = len(glyphs)
        for px in SIZES:
            font = ImageFont.truetype(str(path), px)
            cap = render(font, ord('H'), px)
            xh = render(font, ord('x'), px)
            cap_h = cap[1].height if cap else px
            x_h = xh[1].height if xh else px // 2
            for label, renders in charset:
                seen = set()
                for cp in renders:
                    if cp not in cps:
                        continue
                    r = render(font, cp, px)
                    if r is None:
                        continue
                    top, crop = r
                    key = (top, crop.size, crop.tobytes())
                    if key in seen:
                        continue
                    seen.add(key)
                    glyphs.append((label, fi, px, top, crop.width, crop.height, cap_h, x_h, crop.tobytes()))
        print(f"  {path.name:28s} {len(glyphs) - n0:6d} glyphs")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
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
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {len(glyphs)} glyphs, {len(charset)} labels, {len(names)} fonts)")


if __name__ == "__main__":
    main()
