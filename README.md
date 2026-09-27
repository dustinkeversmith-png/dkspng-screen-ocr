# dks — deterministic screen segmentation + OCR (C++17, header-only)

No neural weights, no dependencies, bit-for-bit deterministic. Two stages:

1. **Screen partitioning** – colour frame → polarity-free edge map → run-based two-row union-find CCL →
   glyph atoms / icons / images / containers → words → lines → paragraph blocks → containment forest.
2. **Glyph recognition** – text crop → coverage-normalised ink → CCL → atoms → over-segmentation →
   recognition-driven DP against a rendered font atlas (exact k-NN), with a fitted baseline / cap-height
   model deciding case and punctuation.

```
include/dks/
  core/      geometry.hpp  image.hpp  utf8.hpp
  imgproc/   binarize.hpp            edge_mask(), otsu, extract_ink() (polarity, shadow mode, 50% coverage)
  segment/   ccl.hpp                 two-row run CCL, runs kept per component (exact masks on demand)
             hierarchy.hpp           containment forest (grid point-location, not O(N^2))
             layout.hpp              analyze_layout(): text words/lines/blocks, icons, images, containers
  ocr/       features.hpp            16x16 area-resampled bitmap + log-aspect + holes + components (Euler)
             simd_metric.hpp         AVX2 / SSE2 / scalar SAD kernels (identical integer results)
             glyph_cut.hpp           glyph hypotheses = (component, column window) sources
             atlas.hpp               loads data/ui_fonts.dksa, same feature pipeline as live glyphs
             classifier.hpp          exact top-k distinct characters (filter&refine | VP-tree), memo cache,
                                     cheap copies that share the template index (one per thread)
             recognizer.hpp          two-pass DP word/line recogniser + text-evidence stats
  pipeline.hpp                       ScreenReader: layout + OCR pass over every box + text verification
  eval/      metrics.hpp             IoU matching, split/merge, containment depth, CER / EM
  platform/  win32.hpp               (opt-in) GDI screen capture, WIC image decode/encode
tools/       fetch_datasets.py       capped (<=50 MB/dataset) downloader + normaliser (+ local wallpaper negatives)
             build_atlas.py          renders the glyph atlas from system fonts
             make_synth_ocr.py       "rendered font grid" test set (seen + held-out fonts)
             make_report.py          OCR-over-every-box HTML report (report_template.html)
bench/       bench_segment.cpp  bench_ocr.cpp  screen_ocr.cpp (live)  ocr_debug.cpp
tests/       test_core.cpp
results/     outputs of the runs quoted below
```

## Quick start

`CMakePresets.json` pins the MSYS2 MinGW64 toolchain (`C:/msys64/mingw64/bin/gcc.exe` / `g++.exe`) and
Ninja, so a stub compiler earlier on `PATH` is never picked up.

```bash
python tools/fetch_datasets.py          # zenodo, webui, icdar_bd (each capped at 50 MB) + negatives
python tools/build_atlas.py             # data/ui_fonts.dksa (45 fonts x 12 sizes x ~160 chars)
python tools/make_synth_ocr.py          # datasets/synth/{seen,unseen}
cmake --preset mingw64 && cmake --build --preset mingw64 && ctest --preset mingw64
build/bench_segment                     # Zenodo + WebUI bounding / grouping / hierarchy (layout + OCR verify)
build/bench_segment --only negatives    # false boxes on 28 stock wallpapers (no UI, no text)
build/bench_ocr                         # synth, ICDAR Born-Digital, WebUI text
build/screen_ocr --frames 3 --out overlay.png --tsv boxes.tsv   # live desktop
python tools/make_report.py --live      # HTML report of every box's reading (report/ocr_report.html)
```

```cpp
#include "dks/dks.hpp"
#include "dks/pipeline.hpp"
dks::ocr::Atlas atlas;  atlas.load("data/ui_fonts.dksa");  atlas.prune(3.0f);
dks::ocr::Classifier cls(atlas);
dks::ScreenReader reader(cls);          // threads = hardware concurrency; output is thread-count independent
reader.enable_memo(true);
dks::ColorView frame{bgra, w, h, stride, dks::PixelFormat::BGRA32};
dks::ScreenRead r = reader.read(frame);
for (auto& e : r.elements)             // every box: geometry, kind, depth, reading, confidence
    if (e.is_text) use(e.element.bbox, e.text, e.confidence);
```

## Datasets (what actually works)

| dataset | source used | size | notes |
|---|---|---|---|
| Desktop UI Detection (Zenodo 10822752) | `Test Desktop UI Detection Dataset.zip` | 20 MB, 36 screenshots, 6053 boxes | LabelMe JSON; 28 classes; **no text transcriptions** |
| WebUI | HF datasets-server row API on `biglab/webui-test-elements` (strided, 1 viewport per page) + `js0nwu/webui` GitHub sample | 50 MB cap, 322 pages | `biglab/webui-all` is a ~550 GB split zip — a capped partial download can't be unzipped. Row API rate-limits (429) → fetcher backs off + resumes |
| ICDAR 2013 Robust Reading Ch.1 Born-Digital (train) | HF mirror `Berzerker/born_digital_images_dataset` | 22 MB, 410 images, 4195 words | official portal needs login. Mirror stores boxes as **integer percentages** → boxes are widened to full percent cells; the recogniser trims clipped neighbours |

Links from the original brief that did **not** work: `github.com/ocr-matrix/icdar-born-digital-subset` (404);
`HF biglab/ICDAR-2011`-style mirrors are signature-verification data; `MiXaiLL76/ICDAR2013_OCR` is the
scene-text (Ch.2) split. Text GT for real UI OCR comes from the WebUI sample's accessibility tree.

## Results (full sets, `results/*.txt`)

### Deterministic OCR (`bench_ocr`)

| set | n | CER | CER (case-insens.) | exact match | EM (ci) |
|---|---:|---:|---:|---:|---:|
| synth/seen — atlas fonts, unseen sizes / colours | 600 | **0.059** | 0.045 | **0.763** | 0.807 |
| synth/unseen — 10 held-out fonts never in the atlas | 600 | **0.103** | 0.085 | **0.653** | 0.707 |
| ICDAR 2013 Born-Digital word crops | 4195 | **0.397** | 0.379 | **0.391** | 0.409 |
| WebUI real browser text lines (lossy WebP) | 84 | **0.033** | 0.024 | **0.810** | 0.833 |

(synth/seen was regenerated with the 45-font atlas list, which added script and display faces, so it is
not comparable with the earlier 0.056.) Ablation, shape-only nearest template: CER 0.125 / 0.278 / 0.676 / 0.231.

### Screen partitioning + OCR verification (`bench_segment`)

Zenodo Desktop UI (36 native Windows screenshots; GT groups: Text | Icon+WebIcon | Image | every widget/panel = Container).
"Layout" is geometry only; "+OCR" runs the recogniser over every box, drops Text boxes whose reading does
not look like text and promotes Icon boxes whose reading does.

| element | n_gt | P@.50 layout | P@.50 +OCR | R@.50 layout | R@.50 +OCR | F1@.50 +OCR |
|---|---:|---:|---:|---:|---:|---:|
| Text (lines) | 2384 | 0.482 | **0.803** | 0.805 | 0.780 | **0.792** |
| Icon | 2618 | 0.796 | 0.797 | 0.794 | 0.787 | 0.792 |
| Container | 915 | 0.504 | 0.504 | 0.540 | 0.540 | 0.521 |
| Image | 136 | 0.183 | 0.183 | 0.147 | 0.147 | 0.163 |
| **All (class-agnostic)** | 6053 | 0.640 | **0.784** | 0.813 | 0.775 | **0.780** |

Previous round (before gating/gutter/hysteresis/verification): All F1 0.695, Text P 0.511, Container R 0.415.
Text-line grouping with verification: 80.0 % one-to-one, split 5.4 %, merge 1.0 %.

False positives on 28 stock Windows wallpapers (no UI, no text), per image: Text 0.1, Icon 4.3, Container 0.8,
Image 2.2 (several Spotlight wallpapers are genuine photos). The same set gave ~217 Text and ~27 Icon boxes
per image before this round.

Speed (12 threads, per-thread memo; output byte-identical to 1 thread): 1920×1080 screenshot, layout + OCR of
every box: 569 ms single-threaded cold → 138–160 ms cold → **23 ms** on repeated frames.

WebUI (323 pages): with verification, ink-tight Text P 0.308 / R 0.281. WebUI "StaticText" boxes are DOM
element boxes (full-width `<h1>`s, whole `<p>`s, inline `<a>` fragments) and the GT omits most visible text,
so Zenodo is the meaningful bounding benchmark.

### What was tried in the improvement round

| change | result | kept |
|---|---|---|
| ink-luma veto on word / line merges (\|Δ\| > 35) | Zenodo Text R .763→.805, Icon R .759→.808, line merge 2.5 %→0.8 % | yes |
| column-gutter veto (blank band ±2 line heights, only for gaps > 0.6 h) | small gain; a gutter test on the pair's own rows alone would forbid every word space | yes |
| hysteresis edges (24 / 10) for flat-region containers | Container R .415→.543 | yes |
| flat container must be colour-uniform (luma σ ≤ 6) | wallpaper containers 10→0, Zenodo unchanged | yes |
| OCR verification of every box | Text P .48→.80, wallpaper text 5641→4 | yes |
| icon saliency (contrast ≥ 40, calm 3 px surround σ ≤ 12) | keeps 97 % of true icons, removes 91 % of wallpaper icons | yes |
| image region must fill ≥ 35 % of its cell bbox | removes busy-cell chains along antialiased curves, Zenodo unchanged | yes |
| shadow mode in `extract_ink` (3 luminance levels, sharper side = ink) | ICDAR CER .413→.397; desktop icon labels become readable | yes |
| threaded ScreenReader (shared index, static striping for memo locality) | 4× cold, 23 ms warm | yes |
| AVX2 SAD + early abandon | exact, but only ~1.2k of 60k templates reach a full SAD: ~0 % speed change | kept (harmless) |
| fused 16-bit integer bound pass | slower (260 vs 200 µs/query) | reverted |
| projection margin trim before Otsu | ICDAR CER .413→.561 (strips the clean border the polarity vote needs) | off |
| Sauvola local threshold (k 0.2 / 0.1) | ICDAR CER .413→.454 / .461, all sets worse | off |
| stroke-width side choice in shadow mode | fixes desktop labels, ICDAR .400→.439 | replaced by sharpness |

Parameters were tuned on the first 300 samples of each OCR set and on Zenodo itself; treat the numbers as
development-set results.

## Design notes vs. the original sketch

* **Binarisation.** Whole-screen segmentation uses a polarity-free *edge map* (any channel step > 28 to the
  right/bottom neighbour); flat UI disappears, text/borders/icons remain, dark and light themes behave the
  same. Recognition uses per-crop ink: Otsu only to find background/ink *levels*, then a **50 % coverage**
  threshold — Otsu's own cut drifts with the crop histogram, which moved antialiased columns in/out of small
  glyphs and was the single largest early error source.
* **CCL.** Classic two-row run-length union-find; no W×H label image; runs are kept grouped per component
  so any component (or a column window of it) can be re-rasterised exactly.
* **Hierarchy.** A uniform-grid point-location index instead of an interval tree: every container of B
  covers B's top-left pixel, so one cell is scanned (largest-first ordering makes the last hit the parent).
* **Euler number.** For 8-connected foreground the quad formula is χ = (C1 − C3 − **2**·CD)/4
  ((C1 − C3 + 2·CD)/4 is the 4-connected variant), and the glyph needs a zero frame so boundary quads count.
  χ alone conflates "i" (2 parts) with "o" (1 hole) → the feature uses **components** and **holes = C − χ**
  separately; the component count is what stops the DP from reading "st"/"re" as "m".
* **Descriptor.** An 8×8 hash was too coarse for 6–10 px UI glyphs; the descriptor is an aspect-preserving
  16×16 area-splat bitmap ([1 2 1]² smoothed, peak-normalised) + log2 aspect + holes + components, compared
  with L1 (a true metric → exact pruning is valid). The coarse 4×4 block sums are an exact lower bound used to
  skip 98 % of full comparisons.
* **"Hash lookup".** Exact hashing of glyph features is used as a *memo* (same raster → same answer); it is
  what makes repeated live frames cheap. A per-character lookup in ~40 ns is not realistic for exact k-NN
  over tens of thousands of templates; ~0.2 ms is what this achieves cold.
* **Recognition.** A one-glyph-at-a-time classifier is not enough: touching glyphs, i-dots, kerning and
  case twins need context. The recogniser scores every segmentation with a DP whose per-glyph cost is
  shape distance × (width / text height) + a character prior + a baseline/cap-height placement error, fitted
  robustly from the unambiguous glyphs of the same line. A deterministic per-word consistency pass fixes
  O/0, l/I/1, S/5 and case twins by the word's other letters (no dictionary).

## Limitations / next steps

* ICDAR Born-Digital is dominated by decorative display faces and JPEG artefacts; with system fonts only
  the atlas can't cover them (see the unseen-font gap). Adding more faces is the cheapest win.
* Image and Container detection are the weak classes; the container rule only knows outlined rectangles
  and flat regions, not gradient panels.
* Line OCR expects single-line crops; the layout's line boxes are used for that.
* The Windows platform layer is the only one provided; the library itself is platform-free.
* `deterministc_screen_ocr.hpp` (the original sketch) and `datasets/fetch.py` are left untouched;
  `tools/fetch_datasets.py` supersedes the latter.
