# Running and inspecting the modular pipeline

The pipeline runs stages over one shared `AnalysisContext`. Each stage reads what earlier stages wrote and adds
its own results; nothing changes the core layout or OCR types.

```
frame ─► CursorMaskStage ─► LayoutStage ─► TextReadStage ─► DetectionStage ─► LatexStage
          (live only)        Layer 0         Layer 2          Layer 1 packs     Layer 2/3
          inpaints the       elements,       {"text", ...}    {"widget", ...}   ctx.regions with
          pointer rect       kinds, depth                     {"widget.state"}  {"latex", ...}
```

All commands below run from the repository root after building:

```bash
cmake --preset mingw64 && cmake --build --preset mingw64
```

## 1. Command line (`screen_pipeline`)

Live virtual desktop, with the trained packs in `data/packs` and the LaTeX stage:

```bash
build/screen_pipeline --packs data/packs --latex
```

The same on a saved screenshot (the cursor stage is skipped for files), writing every box and its tags to a TSV:

```bash
build/screen_pipeline --image "datasets/zenodo/Fullscreen/CRM/V1_image0004.png" --packs data/packs --latex --tsv boxes.tsv
```

A formula image, reading every text box as math:

```bash
build/screen_pipeline --image datasets/im2latex/test/<name>.png --latex-all --no-detect
```

Useful switches while debugging:

| flag | effect |
|---|---|
| `--no-text` | drop the OCR stage (layout + packs only; fastest) |
| `--no-detect` | drop the pack stage |
| `--no-rule-packs` | don't add the training-free `CheckStatePack` fallback |
| `--no-cursor` | live capture without cursor masking |
| `--packs DIR` | load every `*.dkpk` in DIR; files with `state` in the name only run on boxes tagged `ui.checkbox` / `ui.radio` |

Output: the stage list, per-stage time, tag counts, and each LaTeX region:

```
stages: layout text detection latex
  layout           11.7 ms
  text             55.9 ms
  ...
  widget=icon.menu                 4
  widget.state=checked             1
  region [0,0 169x38] latex = \frac { d } { d B } C _ { 1 } = ...
```

The TSV has one line per layout element: `x y w h kind depth` followed by `key=value` for each tag.

For speed numbers, `build/bench_perf [--threads N]` reads the 36 Zenodo screens cold and then again warm, prints
where the time goes, and prints a checksum of every reading (it must not change for an exact optimisation).

## 2. From C++

`examples/pipeline_example.cpp` is a complete program (built as `build/pipeline_example`). The essential parts:

```cpp
#include "dks/pipeline/core_stages.hpp"
#include "dks/detection/detection_pack.hpp"
#include "dks/detection/rule_packs.hpp"
#include "dks/latex/latex_stage.hpp"
#include "dks/platform/win32.hpp"

// Recognisers own an atlas + classifier. Keep the atlases alive as long as the pipeline.
dks::ocr::Atlas ui_atlas, math_atlas;
ui_atlas.load("data/ui_fonts.dksa");     ui_atlas.prune(3.0f);
math_atlas.load("data/math_fonts.dksa"); math_atlas.prune(3.0f);
const dks::ocr::Classifier ui_cls(ui_atlas);
auto reader = std::make_shared<dks::ScreenReader>(ui_cls);   // multi-threaded, output thread-count independent
reader->enable_memo(true);                                   // glyph memo + box cache for repeated frames

// Packs: a type pack, then a state pack scoped to checkbox / radio boxes.
auto packs = std::make_shared<dks::detect::PackRegistry>();
packs->add(dks::detect::TemplatePack::load("data/packs/rico_widget_type.dkpk"));
dks::detect::PackScope on_checkboxes;
on_checkboxes.require_key = "widget";
on_checkboxes.require_any = {"ui.checkbox", "ui.radio"};
packs->add(dks::detect::TemplatePack::load("data/packs/check_state.dkpk", on_checkboxes));

auto math_cls = std::make_shared<dks::ocr::Classifier>(math_atlas);
dks::latex::MathPrior prior;
prior.load("data/math_prior.tsv");

dks::Pipeline pipe;
pipe.add(std::make_shared<dks::CursorMaskStage>([](dks::Rect& r) { return dks::win32::cursor_rect(r); }))
    .add(std::make_shared<dks::LayoutStage>())
    .add(std::make_shared<dks::TextReadStage>(reader))
    .add(std::make_shared<dks::detect::DetectionStage>(packs))
    .add(std::make_shared<dks::latex::LatexStage>(math_cls, dks::latex::LatexStageParams{},
                                                 dks::latex::MathReadParams{}, dks::latex::ParseParams{}, prior));

dks::win32::Frame frame = dks::win32::capture_screen();
dks::AnalysisContext ctx = pipe.run(frame.view());
```

Reading the results:

```cpp
for (size_t i = 0; i < ctx.layout.elements.size(); ++i) {
    const dks::Element& e = ctx.layout.elements[i];            // bbox, kind, depth, parent
    if (const auto* t = ctx.find_tag(i, "text"))         use_text(e.bbox, t->value, t->score);
    if (const auto* w = ctx.find_tag(i, "widget"))       use_widget(e.bbox, w->value);      // e.g. ui.checkbox
    if (const auto* s = ctx.find_tag(i, "widget.state")) use_state(e.bbox, s->value);       // checked / unchecked
}
for (const auto& r : ctx.regions)                               // formulas can span several elements
    for (const auto& t : r.tags)
        if (t.key == "latex") use_formula(r.box, t.value);      // \frac { a } { b } ...
for (const auto& t : ctx.timings) log(t.stage, t.ms);
```

Changing the pipeline at runtime:

```cpp
packs->remove("check_state.knn");   // next run: no widget.state tags
pipe.set_enabled("latex", false);   // skip a stage
pipe.remove("detection");           // or remove it
pipe.add(my_stage, "text");         // insert a custom IPipelineStage before the "text" stage
```

A custom stage implements `name()` and `process(AnalysisContext&)`, and writes results with `ctx.tag(i, key, value,
score, name())` or by appending to `ctx.regions`.

## 3. Debugging checklist

* No text tags: check `ctx.reads[i].text` and `ctx.reads[i].confidence`. The text stage reads every Text and Icon box;
  a box is tagged only when the reading passes verification (3+ glyphs, mostly letters/digits, good matches, one
  baseline). Containers and images are not read unless `TextVerifyParams::read_containers / read_images` is set.
* A pack never fires: its `PackScope` decides element kinds, size range (default 8 to 160 px) and any required tag.
  `DetectionStage` keeps a match only at confidence 0.5 or higher.
* No LaTeX on a normal screen: the stage is deliberately conservative (about 23% of formula images claimed, under 0.1
  false claims per screen). For content you know is math, set `LatexStageParams::claim_all_text`.
* Odd glyph decisions: `build/ocr_debug <image> x y w h --trace` prints every glyph hypothesis and its candidates.
