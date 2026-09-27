// Gating test for the LaTeX pipeline stage (layout -> text -> latex, no claim_all_text):
//   * positives: im2latex formula images on a page margin — share of images where math is claimed
//   * negatives: Zenodo desktop screenshots and stock wallpapers — boxes claimed as LaTeX per image
//
//   bench_latex_stage [--data datasets] [--n 300] [--dump evidence.tsv]
#include <cstdio>
#include <cstring>
#include <fstream>

#include "common.hpp"
#include "dks/latex/latex_stage.hpp"
#include "dks/pipeline/core_stages.hpp"

using namespace dks;

int main(int argc, char** argv) {
    const std::string root = bench::arg_value(argc, argv, "--data", "datasets");
    const size_t N = size_t(std::stoul(bench::arg_value(argc, argv, "--n", "300")));
    const std::string dump = bench::arg_value(argc, argv, "--dump", "");
    ocr::Atlas ui, math;
    if (!ui.load("data/ui_fonts.dksa") || !math.load("data/math_fonts.dksa")) { std::fprintf(stderr, "atlases missing\n"); return 1; }
    ui.prune(3.0f);
    math.prune(3.0f);
    ocr::Classifier ui_cls(ui);
    auto reader = std::make_shared<ScreenReader>(ui_cls);
    reader->enable_memo(true);
    auto math_cls = std::make_shared<ocr::Classifier>(math);
    latex::MathPrior prior;
    prior.load("data/math_prior.tsv");
    auto stage = std::make_shared<latex::LatexStage>(math_cls, latex::LatexStageParams{}, latex::MathReadParams{},
                                                     latex::ParseParams{}, prior);

    Pipeline pipe;  // Layer 0 + text; the LaTeX stage is applied by hand below to record its evidence
    pipe.add(std::make_shared<LayoutStage>()).add(std::make_shared<TextReadStage>(reader));
    std::ofstream out;
    if (!dump.empty()) out.open(dump, std::ios::binary);

    auto run = [&](const std::string& ds, const std::string& filter, size_t limit, bool positive) {
        size_t imgs = 0, hit = 0, claims = 0;
        for (const auto& s : bench::load_manifest(root + "/" + ds)) {
            if (imgs >= limit) break;
            if (!filter.empty() && s.image.find(filter) == std::string::npos) continue;
            win32::Frame f = win32::load_image(s.image);
            if (f.empty()) continue;
            win32::Frame page;  // formula crops are tight: give them the margin they would have on a page
            if (positive) {
                const int m = 24;
                page.width = f.width + 2 * m, page.height = f.height + 2 * m;
                page.pixels.assign(size_t(page.width) * page.height * 4, 255);
                for (int y = 0; y < f.height; ++y)
                    std::memcpy(&page.pixels[(size_t(y + m) * page.width + m) * 4], &f.pixels[size_t(y) * f.width * 4], size_t(f.width) * 4);
            }
            AnalysisContext ctx = pipe.run(positive ? page.view() : f.view());
            const Gray8& luma = ctx.ensure_luma();
            size_t n = 0;
            for (const auto& g : stage->regions(ctx)) {
                const Rect box = g.box;
                const auto ev = latex::math_evidence(stage->reader(), luma.cview().sub(box));
                n += stage->accept(ev);
                if (out)
                    out << ds << '\t' << int(positive) << '\t' << imgs << '\t' << ev.symbols << '\t' << ev.mean_dist << '\t'
                        << ev.rare << '\t' << ev.math_marks << '\t' << box.w << '\t' << box.h << '\n';
            }
            ++imgs, claims += n, hit += n > 0;
        }
        std::printf("  %-12s %4zu images   with >=1 LaTeX claim: %5.1f%%   claims/image %.2f\n", ds.c_str(), imgs,
                    imgs ? 100.0 * double(hit) / double(imgs) : 0.0, imgs ? double(claims) / double(imgs) : 0.0);
    };
    std::printf("LaTeX stage gating (layout + text + latex):\n");
    run("im2latex", "/test/", N, true);
    run("zenodo", "", 1000, false);
    run("negatives", "", 1000, false);
    return 0;
}
