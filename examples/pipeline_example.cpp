// Building and inspecting the modular pipeline from C++ (see docs/PIPELINE.md).
//
//   build/pipeline_example                       live virtual desktop
//   build/pipeline_example shot.png              a saved screenshot
//   build/pipeline_example formula.png --math    treat text as math (LatexStage claim_all_text)
#include <cstdio>
#include <memory>
#include <string>

#include "dks/detection/detection_pack.hpp"
#include "dks/detection/rule_packs.hpp"
#include "dks/latex/latex_stage.hpp"
#include "dks/pipeline/core_stages.hpp"
#include "dks/platform/win32.hpp"

int main(int argc, char** argv) {
    SetProcessDPIAware();
    const std::string image = argc > 1 && argv[1][0] != '-' ? argv[1] : "";
    const bool math_page = argc > 2 && std::string(argv[2]) == "--math";

    // ---- Layer 2 recognisers: each owns its atlas + classifier. The atlases must outlive the pipeline.
    dks::ocr::Atlas ui_atlas, math_atlas;
    if (!ui_atlas.load("data/ui_fonts.dksa") || !math_atlas.load("data/math_fonts.dksa")) {
        std::fprintf(stderr, "run from the repo root (needs data/ui_fonts.dksa and data/math_fonts.dksa)\n");
        return 1;
    }
    ui_atlas.prune(3.0f);
    math_atlas.prune(3.0f);
    const dks::ocr::Classifier ui_cls(ui_atlas);
    auto reader = std::make_shared<dks::ScreenReader>(ui_cls);  // threads = all cores, output identical
    reader->enable_memo(true);                                   // glyph memo + box cache (repeated frames)

    // ---- Layer 1 packs: a widget-type pack, then a state pack that only runs on checkbox / radio boxes.
    auto packs = std::make_shared<dks::detect::PackRegistry>();
    if (auto type = dks::detect::TemplatePack::load("data/packs/rico_widget_type.dkpk")) packs->add(type);
    dks::detect::PackScope on_checkboxes;
    on_checkboxes.require_key = "widget";
    on_checkboxes.require_any = {"ui.checkbox", "ui.radio"};
    if (auto state = dks::detect::TemplatePack::load("data/packs/check_state.dkpk", on_checkboxes)) packs->add(state);
    else packs->add(std::make_shared<dks::detect::CheckStatePack>(on_checkboxes));  // training-free fallback

    // ---- LaTeX: separate stage, own atlas + prior.
    auto math_cls = std::make_shared<dks::ocr::Classifier>(math_atlas);
    dks::latex::MathPrior prior;
    prior.load("data/math_prior.tsv");
    dks::latex::LatexStageParams lp;
    lp.claim_all_text = math_page;

    // ---- Assemble. Order matters only through tags: DetectionStage after LayoutStage, LatexStage after
    //      TextReadStage (it skips boxes the text stage read as one-baseline text).
    dks::Pipeline pipe;
    if (image.empty()) pipe.add(std::make_shared<dks::CursorMaskStage>([](dks::Rect& r) { return dks::win32::cursor_rect(r); }));
    pipe.add(std::make_shared<dks::LayoutStage>())
        .add(std::make_shared<dks::TextReadStage>(reader))
        .add(std::make_shared<dks::detect::DetectionStage>(packs))
        .add(std::make_shared<dks::latex::LatexStage>(math_cls, lp, dks::latex::MathReadParams{}, dks::latex::ParseParams{}, prior));

    dks::win32::Frame frame = image.empty() ? dks::win32::capture_screen() : dks::win32::load_image(image);
    if (frame.empty()) { std::fprintf(stderr, "no frame\n"); return 1; }
    const dks::AnalysisContext ctx = pipe.run(frame.view());

    // ---- Inspect.
    std::printf("stages:");
    for (const auto& t : ctx.timings) std::printf("  %s %.1f ms", t.stage.c_str(), t.ms);
    std::printf("\n%zu layout elements, %zu exclusions (cursor)\n\n", ctx.layout.elements.size(), ctx.exclusions.size());

    for (size_t i = 0; i < ctx.layout.elements.size(); ++i) {
        const dks::Element& e = ctx.layout.elements[i];
        const dks::SemanticTag* text = ctx.find_tag(i, "text");
        const dks::SemanticTag* widget = ctx.find_tag(i, "widget");
        const dks::SemanticTag* state = ctx.find_tag(i, "widget.state");
        if (!text && !widget) continue;
        std::printf("[%4d,%4d %4dx%3d] %-9s depth %d", e.bbox.x, e.bbox.y, e.bbox.w, e.bbox.h, dks::kind_name(e.kind), e.depth);
        if (widget) std::printf("  widget=%s (%.2f)", widget->value.c_str(), widget->score);
        if (state) std::printf("  widget.state=%s (%.2f)", state->value.c_str(), state->score);
        if (text) std::printf("  text=\"%s\" (%.2f)", text->value.c_str(), text->score);
        std::printf("\n");
    }
    for (const auto& r : ctx.regions)
        for (const auto& t : r.tags)
            std::printf("region [%d,%d %dx%d] %s=%s (%.2f, from %zu elements)\n", r.box.x, r.box.y, r.box.w, r.box.h, t.key.c_str(),
                        t.value.c_str(), t.score, r.members.size());
    return 0;
}
