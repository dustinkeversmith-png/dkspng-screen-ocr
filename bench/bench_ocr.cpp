// Deterministic OCR benchmark: CER / exact match on
//   synth/seen    rendered words, atlas fonts at unseen sizes / colours
//   synth/unseen  rendered words, fonts the atlas never saw
//   icdar_bd      ICDAR 2013 Born-Digital (Challenge 1) word crops
//   webui         real browser-rendered UI text lines (WebUI sample axtree)
//
//   bench_ocr [--data datasets] [--atlas data/ui_fonts.dksa] [--vptree] [--only icdar_bd]
//             [--dump out.tsv] [--limit N] [--prune 3.0]
#include <cstdio>
#include <fstream>

#include "common.hpp"
#include "dks/dks.hpp"
#include "dks/ocr/ocr.hpp"

using namespace dks;

namespace {

struct Job {
    std::string name, dir;
    bool words_only;  // crops are single words: no space insertion
};

}  // namespace

int main(int argc, char** argv) {
    const std::string root = bench::arg_value(argc, argv, "--data", "datasets");
    const std::string atlas_path = bench::arg_value(argc, argv, "--atlas", "data/ui_fonts.dksa");
    const std::string only = bench::arg_value(argc, argv, "--only", "");
    const std::string dump = bench::arg_value(argc, argv, "--dump", "");
    const size_t limit = size_t(std::stoul(bench::arg_value(argc, argv, "--limit", "1000000")));
    const float prune = std::stof(bench::arg_value(argc, argv, "--prune", "3.0"));

    ocr::smooth_features() = !bench::has_flag(argc, argv, "--no-smooth");
    bench::Timer tl;
    ocr::Atlas atlas;
    std::string err;
    if (!atlas.load(atlas_path, &err)) { std::fprintf(stderr, "atlas: %s\n", err.c_str()); return 1; }
    const double load_ms = tl.ms();
    bench::Timer tp;
    atlas.prune(prune);
    std::printf("atlas: %zu fonts, %zu raw -> %zu templates after prune(%.1f)  [load %.0f ms, prune %.0f ms]\n",
                atlas.fonts.size(), atlas.raw_count, atlas.templates.size(), prune, load_ms, tp.ms());

    const auto mode = bench::has_flag(argc, argv, "--vptree") ? ocr::SearchMode::VPTree : ocr::SearchMode::Sorted;
    bench::Timer tb;
    ocr::Classifier cls(atlas, {}, mode, std::stof(bench::arg_value(argc, argv, "--margin", "15")));
    std::printf("classifier: %s (build %.0f ms)\n", mode == ocr::SearchMode::VPTree ? "vp-tree" : "aspect-sorted", tb.ms());
    ocr::RecognizerParams P;
    P.char_penalty = std::stof(bench::arg_value(argc, argv, "--penalty", std::to_string(P.char_penalty)));
    P.metric_weight = std::stof(bench::arg_value(argc, argv, "--metric", std::to_string(P.metric_weight)));
    P.line_metrics = !bench::has_flag(argc, argv, "--no-metrics");
    P.topk = std::stoi(bench::arg_value(argc, argv, "--topk", std::to_string(P.topk)));
    P.atom_merge_gap = std::stof(bench::arg_value(argc, argv, "--atom-gap", std::to_string(P.atom_merge_gap)));
    P.atom_merge_px = std::stoi(bench::arg_value(argc, argv, "--atom-px", std::to_string(P.atom_merge_px)));
    P.skip_cost = std::stof(bench::arg_value(argc, argv, "--skip", std::to_string(P.skip_cost)));
    P.space_gap = std::stof(bench::arg_value(argc, argv, "--space", std::to_string(P.space_gap)));
    P.harmonize = !bench::has_flag(argc, argv, "--no-harmonize");
    P.height_weight = std::stof(bench::arg_value(argc, argv, "--hw", std::to_string(P.height_weight)));
    P.width_norm = !bench::has_flag(argc, argv, "--no-wn");
    P.ascii_punct = bench::has_flag(argc, argv, "--ascii");
    P.char_penalty = std::stof(bench::arg_value(argc, argv, "--penalty", std::to_string(P.char_penalty)));
    P.ink.trim_margins = bench::has_flag(argc, argv, "--trim-margins");
    P.ink.sauvola = bench::has_flag(argc, argv, "--sauvola");
    P.ink.shadow_mode = !bench::has_flag(argc, argv, "--no-shadow");
    P.ink.shadow_min_dev = std::stoi(bench::arg_value(argc, argv, "--shadow-dev", "40"));
    P.ink.shadow_sharp_ratio = std::stod(bench::arg_value(argc, argv, "--shadow-ratio", "2.0"));
    P.ink.sauvola_k = std::stof(bench::arg_value(argc, argv, "--sauvola-k", "0.2"));
    P.cut_profile = std::stof(bench::arg_value(argc, argv, "--cut", std::to_string(P.cut_profile)));

    std::ofstream out;
    if (!dump.empty()) out.open(dump, std::ios::binary);

    const Job jobs[] = {{"synth/seen", root + "/synth/seen", false},
                        {"synth/unseen", root + "/synth/unseen", false},
                        {"icdar_bd", root + "/icdar_bd", true},
                        {"webui", root + "/webui", false}};
    for (const Job& job : jobs) {
        if (!only.empty() && job.name.find(only) == std::string::npos) continue;
        const auto samples = bench::load_manifest(job.dir);
        eval::TextScore score;
        double ms = 0;
        size_t glyphs = 0, skipped = 0;
        const uint64_t evals0 = cls.distance_evals, bounds0 = cls.bound_evals;
        for (const auto& s : samples) {
            if (score.n >= limit) break;
            if (job.name == "webui" && s.image.find("/sample/") == std::string::npos) continue;  // only sample has text
            win32::Frame img = win32::load_image(s.image);
            if (img.empty()) continue;
            const Gray8 gray = to_luma(img.view());
            for (const auto& g : bench::load_gt(s.gt)) {
                if (g.text.empty() || g.text == "###") continue;
                if (job.name == "webui" && g.box.h > 40) { ++skipped; continue; }  // wrapped multi-line nodes
                const Rect r = g.box.inflate(job.name == "icdar_bd" ? 0 : 2).clip(gray.width(), gray.height());
                if (r.w < 3 || r.h < 5) { ++skipped; continue; }
                ocr::RecognizerParams p = P;
                p.spaces = !job.words_only;
                const ocr::Recognizer rec(cls, p);
                bench::Timer t;
                const ocr::Recognition res = rec.recognize(gray.cview().sub(r));
                ms += t.ms();
                glyphs += res.glyphs.size();
                const std::string pred = res.utf8();
                score.add(g.text, pred);
                if (out) out << job.name << '\t' << s.image << '\t' << r.x << ',' << r.y << ',' << r.w << ',' << r.h
                             << '\t' << g.text << '\t' << pred << '\n';
            }
        }
        const double evals = double(cls.distance_evals - evals0);
        std::printf("\n== %-13s n=%zu (skipped %zu)\n", job.name.c_str(), score.n, skipped);
        std::printf("   CER %.3f  | CER(case-insens.) %.3f  | EM %.3f  | EM(ci) %.3f\n", score.cer(), score.cer_ci(),
                    score.em(), score.em_ci());
        std::printf("   %.3f ms/crop, %.1f us/glyph, %.0f full distance evals/glyph (%.0f bound checks)\n", ms / double(score.n ? score.n : 1),
                    1000.0 * ms / double(glyphs ? glyphs : 1), evals / double(glyphs ? glyphs : 1), double(cls.bound_evals - bounds0) / double(glyphs ? glyphs : 1));
    }
    return 0;
}
