#!/usr/bin/env python3
"""
Fetch + normalise the three evaluation datasets, each under a hard 50 MB byte cap.

    python tools/fetch_datasets.py            # all
    python tools/fetch_datasets.py webui      # one

Output layout (C++-friendly, no JSON parser needed on the C++ side):

    datasets/<name>/manifest.tsv      image_path \t gt_path          (paths relative to datasets/<name>)
    datasets/<name>/.../*.gt.tsv      x \t y \t w \t h \t label \t text   (integer pixels, text escaped)

Sources (all verified reachable 2026-09):
  zenodo   Desktop UI Detection Dataset, Zenodo record 10822752 (test split, ~15 MB zip, LabelMe JSON)
  webui    biglab/webui-test-elements via the HF datasets-server row API (screenshots + DOM element boxes;
           the full biglab/webui-all is a ~550 GB split zip that cannot be partially unpacked)
           + js0nwu/webui GitHub sample (axtree names -> real rendered text GT for OCR)
  icdar_bd ICDAR 2013 Robust Reading Challenge 1 (Born-Digital) training set, 410 images, word boxes + UTF-8
           transcriptions, mirrored as Berzerker/born_digital_images_dataset (22 MB parquet).
           The official portal rrc.cvc.uab.es needs a logged-in session. NOTE: the mirror stores boxes as
           integer *percentages* of image size, so boxes are quantised (the C++ bench refines them on ink).
"""
from __future__ import annotations

import gzip
import io
import json
import math
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

MAX_BYTES = 50 * 1024 * 1024
ROOT = Path(__file__).resolve().parent.parent / "datasets"
UA = {"User-Agent": "screen-ocr-bench/1.0"}


# ----------------------------------------------------------------------------- utils
class Budget:
    def __init__(self, name: str, cap: int = MAX_BYTES):
        self.name, self.cap, self.used = name, cap, 0

    def take(self, n: int) -> bool:
        if self.used + n > self.cap:
            return False
        self.used += n
        return True

    def __str__(self):
        return f"[{self.name}] {self.used / 1e6:.1f} MB / {self.cap / 1e6:.0f} MB"


def http_get(url: str, budget: Budget, retries: int = 3) -> bytes | None:
    """Streamed GET that refuses to exceed the dataset budget. Returns None if it would."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=60) as resp:
                clen = resp.headers.get("Content-Length")
                if clen and budget.used + int(clen) > budget.cap:
                    return None
                buf = io.BytesIO()
                while chunk := resp.read(1 << 16):
                    buf.write(chunk)
                    if budget.used + buf.tell() > budget.cap:
                        return None
                data = buf.getvalue()
                budget.take(len(data))
                return data
        except Exception as e:  # noqa: BLE001 - network flakiness, retry
            if attempt == retries - 1:
                print(f"  ! {url[:90]}: {e}")
                return None
            time.sleep(1.5 * (attempt + 1))
    return None


def get_json(url: str, retries: int = 6):
    """GET JSON with exponential backoff on HTTP 429 (the HF datasets-server rate-limits bursts)."""
    delay = 5.0
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            if e.code != 429 or attempt == retries - 1:
                raise
            wait = float(e.headers.get("Retry-After") or delay)
            print(f"  429, backing off {wait:.0f}s")
            time.sleep(wait)
            delay = min(delay * 2, 120)


def esc(s: str) -> str:
    return s.replace("\\", "\\\\").replace("\t", "\\t").replace("\n", "\\n").replace("\r", "")


def write_gt(path: Path, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for x, y, w, h, label, text in rows:
            f.write(f"{int(x)}\t{int(y)}\t{int(w)}\t{int(h)}\t{esc(label)}\t{esc(text)}\n")


def write_manifest(dirp: Path, pairs):
    with open(dirp / "manifest.tsv", "w", encoding="utf-8", newline="\n") as f:
        for img, gt in pairs:
            f.write(f"{img.relative_to(dirp).as_posix()}\t{gt.relative_to(dirp).as_posix()}\n")
    print(f"  manifest: {len(pairs)} images -> {dirp / 'manifest.tsv'}")


def sniff_ext(b: bytes) -> str:
    if b[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if b[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if b[:4] == b"GIF8":
        return ".gif"
    if b[:4] == b"RIFF" and b[8:12] == b"WEBP":
        return ".webp"
    if b[:2] == b"BM":
        return ".bmp"
    return ".bin"


def image_size(b: bytes):
    from PIL import Image

    with Image.open(io.BytesIO(b)) as im:
        return im.size


# ----------------------------------------------------------------------------- zenodo
ZENODO_URL = "https://zenodo.org/records/10822752/files/Test%20Desktop%20UI%20Detection%20Dataset.zip"


def fetch_zenodo():
    out = ROOT / "zenodo"
    budget = Budget("zenodo")
    if not any(out.rglob("*.json")):
        print(f"  downloading {ZENODO_URL}")
        data = http_get(ZENODO_URL, budget)
        if data is None:
            sys.exit("  zenodo archive exceeds cap or failed")
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            names = zf.namelist()
            top = {n.split("/")[0] for n in names}
            strip = (next(iter(top)) + "/") if len(top) == 1 else ""
            for n in names:
                if n.endswith("/"):
                    continue
                dst = out / n[len(strip):]
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(zf.read(n))
    else:
        print("  already present, normalising only")

    pairs = []
    for js in sorted(out.rglob("*.json")):
        d = json.loads(js.read_text(encoding="utf-8"))
        img = js.with_suffix(".png")
        if not img.exists():
            continue
        W, H = d["imageWidth"], d["imageHeight"]
        rows = []
        for s in d["shapes"]:
            xs = [p[0] for p in s["points"]]
            ys = [p[1] for p in s["points"]]
            x0, y0 = max(0, math.floor(min(xs))), max(0, math.floor(min(ys)))
            x1, y1 = min(W, math.ceil(max(xs))), min(H, math.ceil(max(ys)))
            if x1 - x0 < 1 or y1 - y0 < 1:
                continue
            rows.append((x0, y0, x1 - x0, y1 - y0, s["label"], s.get("text") or ""))
        gt = js.with_suffix(".gt.tsv")
        write_gt(gt, rows)
        pairs.append((img, gt))
    write_manifest(out, pairs)
    print(" ", budget)


# ----------------------------------------------------------------------------- webui
WEBUI_REPO = "biglab/webui-test-elements"
WEBUI_ROWS = "https://datasets-server.huggingface.co/rows"
WEBUI_SAMPLE = "https://raw.githubusercontent.com/js0nwu/webui/main/sample/1656554031731"
WEBUI_SAMPLE_VIEWS = ["default_1280-720", "default_1366-768", "default_1536-864", "default_1920-1080"]


def webui_primary_label(labels):
    if "StaticText" in labels:
        return "Text"
    for want, name in (("button", "Button"), ("link", "Link"), ("img", "Image"), ("textbox", "TextInput"),
                       ("combobox", "Dropdown"), ("checkbox", "Checkbox"), ("heading", "Heading")):
        if want in labels:
            return name
    return labels[0] if labels else "generic"


def fetch_webui(max_pages: int = 400):
    out = ROOT / "webui"
    img_dir = out / "elements"
    img_dir.mkdir(parents=True, exist_ok=True)
    budget = Budget("webui")
    pairs = []

    # 1) GitHub sample: tiny, but carries real text (axtree names) for OCR evaluation.
    sdir = out / "sample"
    sdir.mkdir(parents=True, exist_ok=True)
    for view in WEBUI_SAMPLE_VIEWS:
        q = lambda suf: f"{WEBUI_SAMPLE}/{urllib.parse.quote(view + suf)}"  # noqa: E731
        shot = http_get(q("-screenshot.webp"), budget)
        ax = http_get(q("-axtree.json.gz"), budget)
        box = http_get(q("-box.json.gz"), budget)
        if not (shot and ax and box):
            continue
        img = sdir / f"{view}.webp"
        img.write_bytes(shot)
        W, H = image_size(shot)
        nodes = json.loads(gzip.decompress(ax))["nodes"]
        boxes = json.loads(gzip.decompress(box))
        rows = []
        for n in nodes:
            if n.get("role", {}).get("value") != "StaticText":
                continue
            text = (n.get("name", {}).get("value") or "").replace("\xa0", " ").strip()
            b = boxes.get(str(n.get("backendDOMNodeId")))
            if not text or not b:
                continue
            pts = b["content"]
            x0, y0 = min(p["x"] for p in pts), min(p["y"] for p in pts)
            x1, y1 = max(p["x"] for p in pts), max(p["y"] for p in pts)
            x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
            if x1 - x0 >= 2 and y1 - y0 >= 4:
                rows.append((round(x0), round(y0), round(x1 - x0), round(y1 - y0), "Text", text))
        gt = sdir / f"{view}.gt.tsv"
        write_gt(gt, rows)
        pairs.append((img, gt))
    print(f"  github sample: {len(pairs)} viewports with text GT")

    # 2) HF element split: one viewport per page, strided across the split for diversity.
    total = get_json(f"{WEBUI_ROWS}?dataset={WEBUI_REPO}&config=default&split=train&offset=0&length=1")["num_rows_total"]
    stride = max(1, total // max_pages) | 1  # odd stride -> cycles through the 4 viewport sizes
    n = 0
    for k in range(max_pages):
        off = (k * stride) % total
        done = list(img_dir.glob(f"r{off:06d}_*.gt.tsv"))  # resumable: skip pages already on disk
        if done:
            imgs = [p for p in img_dir.glob(f"r{off:06d}_*") if not p.name.endswith(".gt.tsv")]
            if imgs:
                budget.take(imgs[0].stat().st_size)
                pairs.append((imgs[0], done[0]))
                n += 1
                continue
        time.sleep(0.5)  # stay under the datasets-server burst limit
        try:
            row = get_json(f"{WEBUI_ROWS}?dataset={WEBUI_REPO}&config=default&split=train&offset={off}&length=1")["rows"][0]["row"]
        except Exception as e:  # noqa: BLE001
            print(f"  ! row {off}: {e}")
            continue
        data = http_get(row["image"]["src"], budget)
        if data is None:
            print("  budget reached")
            break
        W, H = row["image"]["width"], row["image"]["height"]
        stem = f"r{off:06d}_{row['key_name']}"
        img = img_dir / (stem + sniff_ext(data))
        img.write_bytes(data)
        rows = []
        for labels, b in zip(row["labels"], row["contentBoxes"]):
            x0, y0, x1, y1 = max(0, b[0]), max(0, b[1]), min(W, b[2]), min(H, b[3])
            if x1 - x0 < 2 or y1 - y0 < 2:
                continue
            rows.append((round(x0), round(y0), round(x1 - x0), round(y1 - y0),
                         webui_primary_label(labels), "|".join(labels)))
        gt = img_dir / (stem + ".gt.tsv")
        write_gt(gt, rows)
        pairs.append((img, gt))
        n += 1
        if n % 25 == 0:
            print(f"  {n} pages  {budget}")
    write_manifest(out, pairs)
    print(" ", budget)


# ----------------------------------------------------------------------------- icdar born-digital
ICDAR_BD_URL = ("https://huggingface.co/datasets/Berzerker/born_digital_images_dataset/resolve/main/"
                "data/output_born_digital_scene_dataset.parquet")


def fetch_icdar_bd():
    import pyarrow.parquet as pq

    out = ROOT / "icdar_bd"
    img_dir = out / "train"
    img_dir.mkdir(parents=True, exist_ok=True)
    budget = Budget("icdar_bd")
    data = http_get(ICDAR_BD_URL, budget)
    if data is None:
        sys.exit("  icdar_bd parquet exceeds cap or failed")
    table = pq.read_table(io.BytesIO(data))
    pairs = []
    for i, rec in enumerate(table.to_pylist()):
        b = rec["image"]["bytes"] if isinstance(rec["image"], dict) else rec["image"]
        W, H = image_size(b)
        img = img_dir / f"img_{i + 1:03d}{sniff_ext(b)}"
        img.write_bytes(b)
        gt_txt = json.loads(rec["output_json_dumpsed"]) if rec["output_json_dumpsed"].startswith('"') \
            else rec["output_json_dumpsed"]
        rows = []
        for line in gt_txt.splitlines():
            parts = line.split(" ", 4)
            if len(parts) < 5:
                continue
            px0, py0, px1, py1 = map(float, parts[:4])
            # Percent-quantised: widen to the full percent cell so the true box is contained.
            x0 = math.floor(px0 * W / 100.0)
            y0 = math.floor(py0 * H / 100.0)
            x1 = min(W, math.ceil((px1 + 1) * W / 100.0))
            y1 = min(H, math.ceil((py1 + 1) * H / 100.0))
            rows.append((x0, y0, x1 - x0, y1 - y0, "Word", parts[4]))
        gt = img.with_suffix(".gt.tsv")
        write_gt(gt, rows)
        pairs.append((img, gt))
    write_manifest(out, pairs)
    print(" ", budget)


# ----------------------------------------------------------------------------- negatives
def fetch_negatives():
    """No-UI / no-text false-positive set: the stock Windows wallpapers (local copy, nothing downloaded)."""
    import shutil
    web = Path(os.environ.get("WINDIR", "C:/Windows")) / "Web"
    out = ROOT / "negatives"
    out.mkdir(parents=True, exist_ok=True)
    srcs = sorted(p for p in web.rglob("*.jpg") if "touchkeyboard" not in str(p).lower())
    pairs = []
    for i, src in enumerate(srcs, 1):
        img = out / f"w{i:02d}.jpg"
        shutil.copyfile(src, img)
        gt = out / f"w{i:02d}.gt.tsv"
        gt.write_text("", encoding="utf-8")
        pairs.append((img, gt))
    write_manifest(out, pairs)

# ----------------------------------------------------------------------------- im2latex-100k
IM2LATEX = "https://huggingface.co/api/datasets/yuntian-deng/im2latex-100k/parquet/default/{split}/0.parquet"


def fetch_im2latex(splits=("test", "val"), cap_mb: int = 100):
    """im2latex-100k (Deng et al. 2017), clean born-digital Computer-Modern renders + normalised LaTeX.
    test (~34 MB) + val (~30 MB) parquet shards, hard-capped at `cap_mb`."""
    import pyarrow.parquet as pq

    out = ROOT / "im2latex"
    budget = Budget("im2latex", cap_mb * 1024 * 1024)
    pairs = []
    for split in splits:
        data = http_get(IM2LATEX.format(split=split), budget)
        if data is None:
            print(f"  {split}: would exceed the cap, skipped")
            continue
        d = out / split
        d.mkdir(parents=True, exist_ok=True)
        table = pq.read_table(io.BytesIO(data), columns=["formula", "filename", "image"])
        for i, rec in enumerate(table.to_pylist()):
            b = rec["image"]["bytes"] if isinstance(rec["image"], dict) else rec["image"]
            stem = Path(rec["filename"]).stem if rec.get("filename") else f"{i:06d}"
            img = d / f"{stem}{sniff_ext(b)}"
            img.write_bytes(b)
            W, H = image_size(b)
            gt = d / f"{stem}.gt.tsv"
            write_gt(gt, [(0, 0, W, H, "Formula", rec["formula"])])
            pairs.append((img, gt))
        print(f"  {split}: {table.num_rows} formulas  {budget}")
    write_manifest(out, pairs)


# ----------------------------------------------------------------------------- UI widget states
CHECKBOX = "https://huggingface.co/api/datasets/meghnagera15/checkbox-cropped-binary/parquet/default/{split}/0.parquet"


def fetch_ui_states(cap_mb: int = 100):
    """Cropped UI checkboxes / radio buttons labelled checked vs unchecked (meghnagera15/checkbox-cropped-binary)."""
    import pyarrow.parquet as pq

    out = ROOT / "ui_states"
    budget = Budget("ui_states", cap_mb * 1024 * 1024)
    pairs = []
    for split in ("train", "validation", "test"):
        data = http_get(CHECKBOX.format(split=split), budget)
        if data is None:
            print(f"  {split}: would exceed the cap, skipped")
            continue
        table = pq.read_table(io.BytesIO(data))
        names = ["checked", "unchecked"]
        d = out / split
        d.mkdir(parents=True, exist_ok=True)
        for i, rec in enumerate(table.to_pylist()):
            b = rec["image"]["bytes"] if isinstance(rec["image"], dict) else rec["image"]
            label = names[rec["label"]] if isinstance(rec["label"], int) else str(rec["label"])
            img = d / f"{i:05d}{sniff_ext(b)}"
            img.write_bytes(b)
            W, H = image_size(b)
            gt = d / f"{i:05d}.gt.tsv"
            write_gt(gt, [(0, 0, W, H, f"ui.state.{label}", "")])
            pairs.append((img, gt))
        print(f"  {split}: {table.num_rows} crops  {budget}")
    write_manifest(out, pairs)


# ----------------------------------------------------------------------------- Rico widget crops
RICO_SEM = ("https://huggingface.co/datasets/creative-graphic-design/Rico/resolve/main/"
            "ui-screenshots-and-hierarchies-with-semantic-annotations/{split}-00000-of-00001.parquet")
RICO_ROWS = "https://datasets-server.huggingface.co/rows"
RICO_LABELS = ["Text", "Image", "Icon", "Text Button", "List Item", "Input", "Background Image", "Card", "Web View",
               "Radio Button", "Drawer", "Checkbox", "Advertisement", "Modal", "Pager Indicator", "Slider",
               "On/Off Switch", "Button Bar", "Toolbar", "Number Stepper", "Multi-Tab", "Date Picker", "Map View",
               "Video", "Bottom Navigation"]
RICO_WIDGETS = {"Checkbox": "ui.checkbox", "Radio Button": "ui.radio", "On/Off Switch": "ui.switch",
                "Slider": "ui.slider", "Pager Indicator": "ui.pager", "Number Stepper": "ui.stepper"}
RICO_ICONS = {"star", "search", "check", "close", "menu", "add", "arrow_backward", "arrow_forward", "settings",
              "favorite", "share", "delete", "edit", "refresh", "home", "info", "more", "play", "pause", "notifications"}


def fetch_rico_widgets(cap_mb: int = 100, per_class: int = 120, scale: float = 1 / 3, splits=("test", "validation")):
    """Rico (Deka et al. 2017) semantic annotations (Liu et al. 2018): component + icon-class labels are read
    from the 13 MB test annotation shard; only screenshots that contain wanted widgets are fetched (row API),
    and the widget crops are saved downscaled by `scale` (1440x2560 phone -> ~480 px wide)."""
    import pyarrow.parquet as pq
    from PIL import Image

    out = ROOT / "rico_widgets"
    out.mkdir(parents=True, exist_ok=True)
    budget = Budget("rico_widgets", cap_mb * 1024 * 1024)
    # Resumable: crops already on disk count against the budget and against the per-class caps.
    taken, pairs = {}, []
    for gt in sorted(out.rglob("*.gt.tsv")):
        img = gt.with_name(gt.name.replace(".gt.tsv", ".png"))
        if img.exists():
            lab = gt.read_text(encoding="utf-8").split("\t")[4]
            taken[lab] = taken.get(lab, 0) + 1
            pairs.append((img, gt))
    done_rows = {p[0].name.split("_")[0] for p in pairs}
    prev = out / ".downloaded_bytes"
    if prev.exists():
        budget.take(int(prev.read_text()))
    for split in splits:
        _rico_split(split, out, budget, taken, pairs, done_rows, per_class, scale)
    prev.write_text(str(budget.used))
    write_manifest(out, pairs)
    print("  per class:", dict(sorted(taken.items())))
    print(" ", budget)


def _rico_split(split, out, budget, taken, pairs, done_rows, per_class, scale):
    import pyarrow.parquet as pq
    from PIL import Image

    sem = http_get(RICO_SEM.format(split=split), budget)
    if sem is None:
        print(f"  {split}: annotation shard exceeds the remaining budget")
        return
    tag = split[0]  # r = test (original naming), v = validation
    tag = "r" if split == "test" else tag
    table = pq.read_table(io.BytesIO(sem), columns=["children"])
    wanted = {}  # row -> [(label, bounds)]
    for row, rec in enumerate(table.column("children").to_pylist()):
        items = []
        for group in rec or []:
            n = len(group.get("component_label") or [])
            rid = group.get("resource_id") or [None] * n
            kl = group.get("klass") or [None] * n
            for lab, b, ic, r_id, k in zip(group.get("component_label") or [], group.get("bounds") or [],
                                           group.get("icon_class") or [None] * n, rid, kl):
                name = RICO_LABELS[lab] if isinstance(lab, int) and lab < len(RICO_LABELS) else None
                if name in RICO_WIDGETS:
                    items.append((RICO_WIDGETS[name], b))
                elif name == "Icon" and ic in RICO_ICONS:
                    items.append((f"icon.{ic}", b))
                elif name == "Input" and "search" in f"{r_id} {k}".lower():
                    items.append(("ui.searchbar", b))
        if items:
            wanted[row] = items
    # Greedy row order that favours rare classes first, deterministic.
    counts = {}
    for items in wanted.values():
        for lab, _ in items:
            counts[lab] = counts.get(lab, 0) + 1
    order = sorted(wanted, key=lambda r: (min(counts[l] for l, _ in wanted[r]), r))
    for row in order:
        if f"{tag}{row:05d}" in done_rows:
            continue
        items = [(l, b) for l, b in wanted[row] if taken.get(l, 0) < per_class]
        if not items:
            continue
        time.sleep(0.4)
        try:
            meta = get_json(f"{RICO_ROWS}?dataset=creative-graphic-design/Rico&config=ui-screenshots-and-view-hierarchies"
                            f"&split={split}&offset={row}&length=1")
            src = meta["rows"][0]["row"]["screenshot"]["src"]
        except Exception as e:  # noqa: BLE001
            print(f"  ! row {row}: {e}")
            continue
        data = http_get(src, budget)
        if data is None:
            print("  budget reached")
            break
        shot = Image.open(io.BytesIO(data)).convert("RGB")
        sx, sy = shot.width / 1440.0, shot.height / 2560.0  # annotation bounds are in 1440x2560 space
        for k, (lab, b) in enumerate(items):
            x0, y0, x1, y1 = b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy
            if x1 - x0 < 12 or y1 - y0 < 12 or x0 < 0 or y0 < 0 or x1 > shot.width or y1 > shot.height:
                continue
            pad = 6
            crop = shot.crop((max(0, x0 - pad), max(0, y0 - pad), min(shot.width, x1 + pad), min(shot.height, y1 + pad)))
            crop = crop.resize((max(8, round(crop.width * scale)), max(8, round(crop.height * scale))), Image.LANCZOS)
            d = out / lab.replace(".", "_")
            d.mkdir(parents=True, exist_ok=True)
            stem = f"{tag}{row:05d}_{k:02d}"
            img = d / f"{stem}.png"
            crop.save(img)
            gt = d / f"{stem}.gt.tsv"
            write_gt(gt, [(0, 0, crop.width, crop.height, lab, "")])
            pairs.append((img, gt))
            taken[lab] = taken.get(lab, 0) + 1
        if len(pairs) % 50 < len(items):
            print(f"  {split}: {len(pairs)} crops  {budget}", flush=True)


FETCHERS = {"zenodo": fetch_zenodo, "webui": fetch_webui, "icdar_bd": fetch_icdar_bd, "negatives": fetch_negatives,
            "im2latex": fetch_im2latex, "ui_states": fetch_ui_states, "rico_widgets": fetch_rico_widgets}

if __name__ == "__main__":
    # Default run = the original three screen datasets + negatives; the pack datasets are opt-in by name.
    wanted = sys.argv[1:] or ["zenodo", "webui", "icdar_bd", "negatives"]
    for name in wanted:
        print(f"== {name}")
        FETCHERS[name]()
