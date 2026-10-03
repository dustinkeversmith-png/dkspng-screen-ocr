#!/usr/bin/env python3
"""
Cut labelled crops out of screenshots into a folder-per-class dataset for `train_pack --classes`.

    python tools/make_pack_dataset.py --boxes boxes.tsv --out datasets/my_widgets [--pad 2]
    python tools/make_pack_dataset.py --labelme shots/ --out datasets/my_widgets --labels Checkbox,Switch

Inputs (either):
  --boxes FILE    TSV, one box per line:  image_path <TAB> x <TAB> y <TAB> w <TAB> h <TAB> label
                  (the repo's *.gt.tsv rows also work if you prefix the image path)
  --labelme DIR   LabelMe JSON files (as in the Zenodo dataset); rectangles and polygons are used,
                  optionally filtered with --labels

Output: OUT/<label>/<image-stem>_<n>.png, ready for
    build/train_pack --classes OUT --name my_widgets --tag-key widget --holdout 5
Crops keep `--pad` pixels of context on each side, like the pipeline's DetectionStage (margin 2).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from PIL import Image


def safe(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", label).strip("_") or "unlabelled"


def crops_from_tsv(path: Path):
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        f = line.split("\t")
        if len(f) < 6:
            raise SystemExit(f"bad line (need image, x, y, w, h, label): {line!r}")
        yield f[0], int(float(f[1])), int(float(f[2])), int(float(f[3])), int(float(f[4])), f[5]


def crops_from_labelme(d: Path, keep: set[str] | None):
    for js in sorted(d.rglob("*.json")):
        data = json.loads(js.read_text(encoding="utf-8"))
        img = js.with_name(data.get("imagePath") or js.with_suffix(".png").name)
        if not img.exists():
            img = js.with_suffix(".png")
        for s in data.get("shapes", []):
            if keep and s["label"] not in keep:
                continue
            xs = [p[0] for p in s["points"]]
            ys = [p[1] for p in s["points"]]
            x0, y0 = int(min(xs)), int(min(ys))
            yield str(img), x0, y0, int(max(xs)) - x0, int(max(ys)) - y0, s["label"]


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--boxes", type=Path)
    src.add_argument("--labelme", type=Path)
    ap.add_argument("--labels", default="", help="comma-separated labels to keep (LabelMe input)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pad", type=int, default=2)
    ap.add_argument("--min-side", type=int, default=6)
    args = ap.parse_args()

    keep = {l.strip() for l in args.labels.split(",") if l.strip()} or None
    rows = crops_from_tsv(args.boxes) if args.boxes else crops_from_labelme(args.labelme, keep)
    cache: dict[str, Image.Image] = {}
    counts: dict[str, int] = {}
    for image, x, y, w, h, label in rows:
        if min(w, h) < args.min_side:
            continue
        if image not in cache:
            cache.clear()  # one screenshot in memory at a time
            cache[image] = Image.open(image).convert("RGB")
        im = cache[image]
        box = (max(0, x - args.pad), max(0, y - args.pad), min(im.width, x + w + args.pad), min(im.height, y + h + args.pad))
        lab = safe(label)
        d = args.out / lab
        d.mkdir(parents=True, exist_ok=True)
        n = counts.get(lab, 0)
        im.crop(box).save(d / f"{Path(image).stem}_{n:04d}.png")
        counts[lab] = n + 1
    for lab, n in sorted(counts.items()):
        print(f"  {lab:28s} {n:5d}")
    print(f"wrote {sum(counts.values())} crops in {len(counts)} classes to {args.out}")


if __name__ == "__main__":
    main()
