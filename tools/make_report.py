#!/usr/bin/env python3
"""
OCR-over-every-box report: runs build/screen_ocr on screenshots, reads every detected box, and writes
one self-contained HTML page (overlays + per-box readings + end-to-end accuracy where text GT exists).

    python tools/make_report.py                       # default dataset samples -> report/ocr_report.html
    python tools/make_report.py --live                # also your live desktop (stays local)
    python tools/make_report.py --image shot.png ...  # any screenshots

Screens with text ground truth (WebUI sample) get end-to-end scoring: each GT text box is matched to
the verified text box with the best IoU (>= 0.3); unmatched GT boxes count as fully missed.
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
EXE = ROOT / "build" / ("screen_ocr.exe" if os.name == "nt" else "screen_ocr")

DEFAULT = [
    ("Zenodo · CRM", "datasets/zenodo/Fullscreen/CRM/V1_image0004.png", None),
    ("Zenodo · File Explorer", "datasets/zenodo/Fullscreen/File Explorer/Captura de pantalla 2024-01-30 102024.png", None),
    ("Zenodo · Mail", "datasets/zenodo/Fullscreen/Mail/Captura de pantalla 2024-01-30 111214.png", None),
    ("Zenodo · Moodle", "datasets/zenodo/Fullscreen/Moodle/Captura de pantalla (145).png", None),
    ("WebUI · signal.org 1920×1080", "datasets/webui/sample/default_1920-1080.webp", "datasets/webui/sample/default_1920-1080.gt.tsv"),
    ("WebUI · signal.org 1280×720", "datasets/webui/sample/default_1280-720.webp", "datasets/webui/sample/default_1280-720.gt.tsv"),
    ("Negative · Windows bloom wallpaper", "C:/Windows/Web/4K/Wallpaper/Windows/img0_1920x1200.jpg", None),
]


def lev(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    return inter / float(aw * ah + bw * bh - inter) if inter else 0.0


def run_one(label, image, gt, workdir, idx, live=False):
    tsv = Path(workdir) / f"{idx}.tsv"
    png = Path(workdir) / f"{idx}.png"
    raw = Path(workdir) / f"{idx}_raw.png"
    cmd = [str(EXE), "--quiet", "--tsv", str(tsv), "--out", str(png)]
    if live:
        cmd += ["--save-frame", str(raw)]
    else:
        cmd += ["--image", image]
    out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    frame_line = next((l for l in out.stdout.splitlines() if l.startswith("frame")), "")
    ms = frame_line.split("layout+OCR ")[1].split(" ms")[0] if "layout+OCR" in frame_line else "?"
    boxes = []
    for line in tsv.read_text(encoding="utf-8", errors="replace").splitlines():
        f = line.split("\t")
        if len(f) < 8:
            continue
        boxes.append(dict(x=int(f[0]), y=int(f[1]), w=int(f[2]), h=int(f[3]), kind=f[4], depth=int(f[5]),
                          text_ok=f[6] == "1", conf=round(float(f[7]), 2), text=f[8] if len(f) > 8 else ""))
    im = Image.open(png).convert("RGB")
    W, H = im.size
    scale = min(1.0, 1400 / W)
    im = im.resize((int(W * scale), int(H * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=82)
    overlay = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

    e2e = None
    if gt:
        gts = []
        for line in Path(ROOT / gt).read_text(encoding="utf-8").splitlines():
            f = line.split("\t")
            if len(f) >= 6 and f[4] == "Text" and f[5].strip() and int(f[3]) <= 40:
                gts.append(((int(f[0]), int(f[1]), int(f[2]), int(f[3])), f[5].strip()))
        preds = [b for b in boxes if b["text_ok"]]
        rows, edits, chars, exact = [], 0, 0, 0
        for box, text in gts:
            best, bi = 0.0, None
            for b in preds:
                v = iou(box, (b["x"], b["y"], b["w"], b["h"]))
                if v > best:
                    best, bi = v, b
            read = bi["text"] if bi is not None and best >= 0.3 else ""
            e = lev(text, read)
            edits += e
            chars += len(text)
            exact += e == 0
            rows.append(dict(gt=text, read=read, iou=round(best, 2), cer=round(e / max(1, len(text)), 2)))
        e2e = dict(n=len(gts), cer=round(edits / max(1, chars), 3), em=round(exact / max(1, len(gts)), 3), rows=rows)

    kinds = {}
    for b in boxes:
        kinds[b["kind"]] = kinds.get(b["kind"], 0) + 1
    return dict(label=label, source=image if not live else "live desktop", size=[W, H], ms=ms, overlay=overlay,
                boxes=boxes, kinds=kinds, verified=sum(b["text_ok"] for b in boxes),
                rejected=sum(1 for b in boxes if b["kind"] == "Text" and not b["text_ok"]), e2e=e2e)


def read_bench():
    """Headline numbers from results/*.txt when present."""
    out = {}
    for name in ("segment.txt", "segment_layout_only.txt", "negatives_wallpapers.txt", "ocr.txt"):
        p = ROOT / "results" / name
        if p.exists():
            out[name] = p.read_text(encoding="utf-8", errors="replace")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", nargs="*", default=None, help="screenshots to read instead of the defaults")
    ap.add_argument("--live", action="store_true", help="also capture and read the live desktop")
    ap.add_argument("--out", default=str(ROOT / "report" / "ocr_report.html"))
    args = ap.parse_args()
    if not EXE.exists():
        sys.exit(f"build first: {EXE} not found (cmake --preset mingw64 && cmake --build --preset mingw64)")
    inputs = [(Path(p).name, p, None) for p in args.image] if args.image else DEFAULT
    screens = []
    with tempfile.TemporaryDirectory() as td:
        for i, (label, image, gt) in enumerate(inputs):
            if not Path(image).is_absolute() and not (ROOT / image).exists():
                print(f"  skip (missing) {image}")
                continue
            print(f"  reading {label}")
            screens.append(run_one(label, str(ROOT / image) if not Path(image).is_absolute() else image, gt, td, i))
        if args.live:
            print("  reading live desktop")
            screens.append(run_one("Live desktop", "", None, td, 999, live=True))
    data = dict(screens=screens, bench=read_bench())
    tpl = (Path(__file__).parent / "report_template.html").read_text(encoding="utf-8")
    page = tpl.replace("/*__DATA__*/null", json.dumps(data, ensure_ascii=False).replace("</", "<\\/"))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {len(screens)} screens)")


if __name__ == "__main__":
    main()
