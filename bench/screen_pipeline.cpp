// Modular pipeline on the live screen or an image. Stages and packs are chosen at runtime.
//
//   screen_pipeline [--image shot.png]             default: cursor_mask, layout, text, detection
//                   [--no-text] [--no-detect] [--no-cursor]
//                   [--latex]                      add the LaTeX stage (data/math_fonts.dksa)
//                   [--latex-all]                  read every Text box as math (formula images)
//                   [--packs data/packs]           load every *.dkpk there as a TemplatePack
//                   [--no-rule-packs]              drop the built-in rule packs
//                   [--tsv out.tsv]                x y w h kind depth tags...
#include <cstdio>
#include <fstream>
#include <map>

#include "common.hpp"
#include "dks/detection/detection_pack.hpp"
#include "dks/detection/rule_packs.hpp"
#include "dks/latex/latex_stage.hpp"
#include "dks/pipeline/core_stages.hpp"

using namespace dks;

int main(int argc, char** argv) {
    SetProcessDPIAware();
    const std::string image = bench::arg_value(argc, argv, "--image", "");
    const std::string tsv = bench::arg_value(argc, argv, "--tsv", "");
    const std::string packs_dir = bench::arg_value(argc, argv, "--packs", "data/packs");

    Pipeline pipe;
    if (image.empty() && !bench::has_flag(argc, argv, "--no-cursor"))
        pipe.add(std::make_shared<CursorMaskStage>([](Rect& r) { return win32::cursor_rect(r); }));
    pipe.add(std::make_shared<LayoutStage>());

    // Layer 2: base text (ui_fonts.dksa)
    ocr::Atlas ui_atlas;
    std::shared_ptr<ocr::Classifier> ui_cls;
    if (!bench::has_flag(argc, argv, "--no-text")) {
        if (!ui_atlas.load("data/ui_fonts.dksa")) { std::fprintf(stderr, "missing data/ui_fonts.dksa\n"); return 1; }
        ui_atlas.prune(3.0f);
        ui_cls = std::make_shared<ocr::Classifier>(ui_atlas);
        auto reader = std::make_shared<ScreenReader>(*ui_cls);
        reader->enable_memo(true);
        pipe.add(std::make_shared<TextReadStage>(reader));
    }

    // Layer 1: detection packs (pluggable)
    auto registry = std::make_shared<detect::PackRegistry>();
    if (!bench::has_flag(argc, argv, "--no-detect")) {
        WIN32_FIND_DATAA fd;
        HANDLE h = FindFirstFileA((packs_dir + "/*.dkpk").c_str(), &fd);
        for (BOOL ok = h != INVALID_HANDLE_VALUE; ok; ok = FindNextFileA(h, &fd)) {
            detect::PackScope scope;
            std::string file = fd.cFileName;
            // State packs only look at boxes a type pack already called a checkbox / radio.
            if (file.find("state") != std::string::npos) scope.require_key = "widget", scope.require_any = {"ui.checkbox", "ui.radio"};
            if (auto p = detect::TemplatePack::load(packs_dir + "/" + file, scope)) {
                std::printf("pack %-24s %zu exemplars, %zu classes -> tag '%s'\n", p->name().c_str(), p->size(),
                            p->classes().size(), p->tag_key().c_str());
                registry->add(p);
            }
        }
        if (h != INVALID_HANDLE_VALUE) FindClose(h);
        if (!bench::has_flag(argc, argv, "--no-rule-packs")) {
            detect::PackScope sc;
            sc.require_key = "widget";
            sc.require_any = {"ui.checkbox", "ui.radio"};
            // rule pack only when no trained state pack is loaded
            bool have_state = false;
            for (const auto& p : registry->packs()) have_state |= p->tag_key() == "widget.state";
            if (!have_state) registry->add(std::make_shared<detect::CheckStatePack>(sc));
        }
        pipe.add(std::make_shared<detect::DetectionStage>(registry));
    }

    // Layer 2/3: LaTeX (math_fonts.dksa), separate from the text stage
    ocr::Atlas math_atlas;
    if (bench::has_flag(argc, argv, "--latex") || bench::has_flag(argc, argv, "--latex-all")) {
        if (!math_atlas.load("data/math_fonts.dksa")) { std::fprintf(stderr, "missing data/math_fonts.dksa\n"); return 1; }
        math_atlas.prune(3.0f);
        auto mc = std::make_shared<ocr::Classifier>(math_atlas);
        latex::LatexStageParams lp;
        lp.claim_all_text = bench::has_flag(argc, argv, "--latex-all");
        latex::MathPrior prior;
        prior.load("data/math_prior.tsv");
        pipe.add(std::make_shared<latex::LatexStage>(mc, lp, latex::MathReadParams{}, latex::ParseParams{}, prior));
    }

    std::printf("stages:");
    for (const auto& n : pipe.names()) std::printf(" %s", n.c_str());
    std::printf("\n");

    win32::Frame frame = image.empty() ? win32::capture_screen() : win32::load_image(image);
    if (frame.empty()) { std::fprintf(stderr, "no frame\n"); return 1; }
    AnalysisContext ctx = pipe.run(frame.view());

    for (const auto& t : ctx.timings) std::printf("  %-12s %8.1f ms\n", t.stage.c_str(), t.ms);
    std::map<std::string, int> counts;
    for (const auto& tags : ctx.tags)
        for (const auto& t : tags)
            if (t.key != "kind") ++counts[t.key + "=" + (t.key == "text" || t.key == "latex" ? std::string("*") : t.value)];
    for (const auto& [k, v] : counts) std::printf("  %-32s %d\n", k.c_str(), v);
    for (const auto& r : ctx.regions)
        for (const auto& t : r.tags)
            std::printf("  region [%d,%d %dx%d] %s = %s\n", r.box.x, r.box.y, r.box.w, r.box.h, t.key.c_str(), t.value.c_str());

    if (!tsv.empty()) {
        std::ofstream o(tsv, std::ios::binary);
        for (size_t i = 0; i < ctx.layout.elements.size(); ++i) {
            const auto& e = ctx.layout.elements[i];
            o << e.bbox.x << '\t' << e.bbox.y << '\t' << e.bbox.w << '\t' << e.bbox.h << '\t' << kind_name(e.kind) << '\t' << e.depth;
            if (i < ctx.tags.size())
                for (const auto& t : ctx.tags[i]) {
                    std::string v = t.value;
                    for (char& c : v)
                        if (c == '\t' || c == '\n') c = ' ';
                    o << '\t' << t.key << '=' << v;
                }
            o << '\n';
        }
    }
    return 0;
}
