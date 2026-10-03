// === C:\Users\Cutie Magic 500\projects\creative\screen-ocr\bench\bench_latex.cpp ===
// im2latex-100k benchmark for the LaTeX stage (MathReader + spatial tree parser).
//
//   bench_latex [--data datasets] [--atlas data/math_fonts.dksa] [--split test] [--limit N] [--dump out.tsv]
#include <cstdio>
#include <fstream>

#include "common.hpp"
#include "dks/eval/latex_metrics.hpp"
#include "dks/latex/spatial_tree_parsing.hpp"

#include <sstream>
#include <unordered_map>

using namespace dks;

int main(int argc, char** argv) {
    const std::string root = bench::arg_value(argc, argv, "--data", "datasets");
    const std::string split = bench::arg_value(argc, argv, "--split", "test");
    const size_t limit = size_t(std::stoul(bench::arg_value(argc, argv, "--limit", "1000000")));
    const std::string dump = bench::arg_value(argc, argv, "--dump", "");

    bench::Timer tl;
    ocr::Atlas atlas;
    std::string err;
    if (!atlas.load(bench::arg_value(argc, argv, "--atlas", "data/math_fonts.dksa"), &err)) {
        std::fprintf(stderr, "math atlas: %s (python tools/build_math_atlas.py)\n", err.c_str());
        return 1;
    }
    atlas.prune(std::stof(bench::arg_value(argc, argv, "--prune", "3.0")));
    ocr::Classifier cls(atlas);
    cls.enable_memo(true);
    latex::MathReadParams mp;
    mp.rescore = !bench::has_flag(argc, argv, "--no-rescore");
    mp.scale_weight = std::stof(bench::arg_value(argc, argv, "--scale-w", std::to_string(mp.scale_weight)));
    // --make-prior: count label frequencies in the GT of --prior-split (never the evaluated split).
    const std::string prior_path = bench::arg_value(argc, argv, "--prior", "data/math_prior.tsv");
    if (bench::has_flag(argc, argv, "--make-prior")) {
        const std::string psplit = bench::arg_value(argc, argv, "--prior-split", "val");
        std::unordered_map<std::string, char32_t> tok2cp;
        for (const auto& t : atlas.templates) tok2cp[latex::tex_of(t.ch)] = t.ch;
        std::unordered_map<char32_t, double> counts;
        size_t n = 0;
        for (const auto& s : bench::load_manifest(root + "/im2latex")) {
            if (s.image.find("/" + psplit + "/") == std::string::npos) continue;
            const auto gt = bench::load_gt(s.gt);
            if (gt.empty()) continue;
            std::istringstream in(gt[0].text);
            std::string w;
            while (in >> w) {
                if (w == "\\to") w = "\\rightarrow";
                if (w == "\\le") w = "\\leq";
                if (w == "\\ge") w = "\\geq";
                if (w == "\\prime") w = "'";
                const auto it = tok2cp.find(w);
                if (it != tok2cp.end()) counts[it->second] += 1;
            }
            ++n;
        }
        std::ofstream o(prior_path, std::ios::binary);
        for (const auto& [c, k] : counts) o << uint32_t(c) << ' ' << k << '\n';
        std::printf("wrote %s from %zu %s formulas (%zu labels)\n", prior_path.c_str(), n, psplit.c_str(), counts.size());
        return 0;
    }
    latex::MathPrior prior;
    if (!bench::has_flag(argc, argv, "--no-prior") && prior.load(prior_path)) std::printf("prior: %s\n", prior_path.c_str());
    mp.prior_weight = std::stof(bench::arg_value(argc, argv, "--prior-w", std::to_string(mp.prior_weight)));
    latex::MathReader reader(cls, mp, prior);
    latex::ParseParams pp;
    pp.script_scale = std::stof(bench::arg_value(argc, argv, "--script-scale", std::to_string(pp.script_scale)));
    std::printf("math atlas: %zu fonts, %zu templates (%.0f ms)\n", atlas.fonts.size(), atlas.templates.size(), tl.ms());

    std::ofstream out;
    if (!dump.empty()) out.open(dump, std::ios::binary);
    eval::LatexScore score;
    double ms = 0;
    size_t syms = 0;
    for (const auto& s : bench::load_manifest(root + "/im2latex")) {
        if (score.n >= limit) break;
        if (s.image.find("/" + split + "/") == std::string::npos) continue;
        const auto gt = bench::load_gt(s.gt);
        if (gt.empty()) continue;
        win32::Frame f = win32::load_image(s.image);
        if (f.empty()) continue;
        const Gray8 g = to_luma(f.view());
        bench::Timer t;
        const auto sy = reader.symbols(g.cview());
        const std::string pred = latex::parse_latex(sy, pp);
        ms += t.ms();
        syms += sy.size();
        score.add(gt[0].text, pred);
        if (bench::has_flag(argc, argv, "--show") && gt[0].text.find(" 1 ") != std::string::npos && score.n <= 40) {
            std::printf("GT  %s\nOUT %s\n", gt[0].text.c_str(), pred.c_str());
            for (const auto& q : sy) {
                if (q.kind != latex::MathSymbol::Kind::Glyph || !(q.ch == U'I' || q.ch == U'l' || q.ch == U'J' || q.ch == U'1')) continue;
                std::printf("   [%d,%d %dx%d]", q.box.x, q.box.y, q.box.w, q.box.h);
                for (int c = 0; c < q.ncand; ++c) {
                    std::string u;
                    utf8_append(u, q.cand[c].ch);
                    const auto& t = atlas.templates[q.cand[c].tmpl];
                    std::printf(" %s:%.1f(%s %d)", u.c_str(), q.cand[c].dist, atlas.fonts[t.font].c_str(), t.px);
                }
                std::printf("\n");
            }
        }
        if (out) out << s.image << '\t' << gt[0].text << '\t' << pred << '\n';
    }
    std::printf("\n== im2latex/%s  n=%zu\n", split.c_str(), score.n);
    std::printf("   token edit distance / |gt| %.3f   exact match %.3f\n", score.ted(), score.em());
    std::printf("   symbols: precision %.3f  recall %.3f  F1 %.3f\n", score.sym_p(), score.sym_r(), score.sym_f1());
    std::printf("   %.2f ms/formula, %.1f symbols/formula\n", ms / double(score.n ? score.n : 1), double(syms) / double(score.n ? score.n : 1));
    return 0;
}

// === C:\Users\Cutie Magic 500\projects\creative\screen-ocr\bench\bench_latex_stage.cpp ===
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

// === C:\Users\Cutie Magic 500\projects\creative\screen-ocr\bench\bench_ocr.cpp ===
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

// === C:\Users\Cutie Magic 500\projects\creative\screen-ocr\bench\bench_packs.cpp ===
// Detection-pack benchmark.
//
//   ui_states     checkbox / radio crops, checked vs unchecked (train+validation -> test)
//                 * TemplatePack (k-NN over ShapeDescriptors) and the training-free CheckStatePack
//   rico_widgets  Rico widget / icon crops, type classification with a screen-disjoint split
//                 (even Rico row -> train, odd row -> test)
//
//   bench_packs [--data datasets] [--save data/packs]   saves the trained packs as .dkpk
#include <cstdio>
#include <map>
#include <string>

#include "common.hpp"
#include "dks/detection/detection_pack.hpp"
#include "dks/detection/rule_packs.hpp"

using namespace dks;
using namespace dks::detect;

namespace {

struct Item {
    std::string label, split;
    Gray8 luma;
};

std::vector<Item> load(const std::string& dir, bool rico) {
    std::vector<Item> out;
    for (const auto& s : bench::load_manifest(dir)) {
        const auto gt = bench::load_gt(s.gt);
        if (gt.empty()) continue;
        win32::Frame f = win32::load_image(s.image);
        if (f.empty()) continue;
        Item it;
        it.label = gt[0].label;
        if (rico) {  // .../r01234_05.png -> Rico row 1234
            const size_t p = s.image.find_last_of("/\\");
            const int row = std::atoi(s.image.c_str() + p + 2);
            it.split = row % 2 == 0 ? "train" : "test";
        } else {
            it.split = s.image.find("/test/") != std::string::npos ? "test" : "train";
        }
        it.luma = to_luma(f.view());
        out.push_back(std::move(it));
    }
    return out;
}

struct Score {
    std::map<std::string, std::map<std::string, int>> conf;  // gt -> pred -> n
    int n = 0, ok = 0;
    void add(const std::string& gt, const std::string& pred) { ++conf[gt][pred], ++n, ok += gt == pred; }
    void print(const char* title) const {
        std::printf("  %-34s accuracy %.3f  (%d / %d)\n", title, n ? double(ok) / n : 0.0, ok, n);
        for (const auto& [gt, row] : conf) {
            int tot = 0, hit = 0;
            for (const auto& [p, c] : row) tot += c, hit += p == gt ? c : 0;
            std::string top;
            for (const auto& [p, c] : row)
                if (p != gt && c * 10 >= tot) top += " " + p + ":" + std::to_string(c);
            std::printf("      %-24s %4d  recall %.2f%s%s\n", gt.c_str(), tot, tot ? double(hit) / tot : 0.0,
                        top.empty() ? "" : "   confused with", top.c_str());
        }
    }
};

}  // namespace

int main(int argc, char** argv) {
    const std::string root = bench::arg_value(argc, argv, "--data", "datasets");
    const std::string save = bench::arg_value(argc, argv, "--save", "data/packs");
    CreateDirectoryA(save.c_str(), nullptr);

    // ---------------------------------------------------------------- checkbox / radio state
    {
        auto items = load(root + "/ui_states", false);
        std::printf("\n== ui_states  (%zu crops)\n", items.size());
        auto pack = std::make_shared<TemplatePack>("check_state.knn", "widget.state");
        for (const auto& it : items)
            if (it.split == "train") pack->add_example(it.label.substr(it.label.rfind('.') + 1), it.luma.cview());
        const CheckStatePack rule;
        Score knn, rl;
        bench::Timer t;
        for (const auto& it : items) {
            if (it.split != "test") continue;
            const std::string gt = it.label.substr(it.label.rfind('.') + 1);
            Patch p;
            p.luma = it.luma.cview();
            p.box = Rect{0, 0, it.luma.width(), it.luma.height()};
            const auto m = pack->classify(p);
            knn.add(gt, m.empty() ? "?" : m.front().tag);
            const auto r = rule.classify(p);
            rl.add(gt, r.empty() ? "?" : r.front().tag);
        }
        std::printf("  %zu exemplars, %.2f ms per crop (both packs)\n", pack->size(), t.ms() / double(knn.n ? knn.n : 1));
        knn.print("TemplatePack k-NN (trained)");
        rl.print("CheckStatePack rule (no training)");
        pack->save(save + "/check_state.dkpk");
    }

    // ---------------------------------------------------------------- Rico widget / icon type
    {
        auto items = load(root + "/rico_widgets", true);
        std::printf("\n== rico_widgets  (%zu crops, screen-disjoint split)\n", items.size());
        if (!items.empty()) {
            auto pack = std::make_shared<TemplatePack>("rico_widget_type.knn", "widget", PackScope{}, std::stoi(bench::arg_value(argc, argv, "--k", "3")));
            for (const auto& it : items)
                if (it.split == "train") pack->add_example(it.label, it.luma.cview());
            Score s, coarse;
            auto group = [](const std::string& l) { return l.rfind("icon.", 0) == 0 ? std::string("icon") : l; };
            for (const auto& it : items) {
                if (it.split != "test") continue;
                Patch p;
                p.luma = it.luma.cview();
                p.box = Rect{0, 0, it.luma.width(), it.luma.height()};
                const auto m = pack->classify(p);
                const std::string pred = m.empty() ? "?" : m.front().tag;
                s.add(it.label, pred);
                coarse.add(group(it.label), group(pred));
            }
            std::printf("  %zu exemplars, %zu classes\n", pack->size(), pack->classes().size());
            s.print("TemplatePack k-NN, fine classes");
            coarse.print("same, icons pooled into 'icon'");
            pack->save(save + "/rico_widget_type.dkpk");
        }
    }
    return 0;
}

// === C:\Users\Cutie Magic 500\projects\creative\screen-ocr\bench\bench_perf.cpp ===
// Performance benchmark for the screen reading path (layout + OCR of every box), with an output
// checksum so exact optimisations can be verified to change nothing.
//
//   bench_perf [--data datasets] [--threads 1] [--prune 3.0] [--no-memo] [--limit N] [--by-kind]
//
// Each screenshot is read twice in a row: "cold" is the first read, "warm" the same frame again (a
// live screen between changes). The memo cache persists across frames, as it would live.
#include <cstdio>
#include <map>

#include "common.hpp"
#include "dks/pipeline.hpp"

using namespace dks;

int main(int argc, char** argv) {
    const std::string root = bench::arg_value(argc, argv, "--data", "datasets");
    const unsigned threads = unsigned(std::stoul(bench::arg_value(argc, argv, "--threads", "1")));
    const float prune = std::stof(bench::arg_value(argc, argv, "--prune", "3.0"));
    const size_t limit = size_t(std::stoul(bench::arg_value(argc, argv, "--limit", "1000")));
    const bool memo = !bench::has_flag(argc, argv, "--no-memo");

    bench::Timer tl;
    ocr::Atlas atlas;
    if (!atlas.load(bench::arg_value(argc, argv, "--atlas", "data/ui_fonts.dksa"))) return 1;
    atlas.prune(prune);
    const double load_ms = tl.ms();
    bench::Timer tb;
    ocr::Classifier cls(atlas, {}, ocr::SearchMode::Sorted, std::stof(bench::arg_value(argc, argv, "--margin", "15")));
    const double build_ms = tb.ms();
    TextVerifyParams vp;
    vp.read_containers = vp.read_images = bench::has_flag(argc, argv, "--read-all");
    ocr::RecognizerParams rp;
    rp.topk = std::stoi(bench::arg_value(argc, argv, "--topk", std::to_string(rp.topk)));
    rp.atom_merge_gap = std::stof(bench::arg_value(argc, argv, "--atom-gap", std::to_string(rp.atom_merge_gap)));
    rp.atom_merge_px = std::stoi(bench::arg_value(argc, argv, "--atom-px", std::to_string(rp.atom_merge_px)));
    ScreenReader reader(cls, {}, rp, vp, threads);
    reader.enable_memo(memo);
    std::printf("atlas %zu templates (load+prune %.0f ms, index %.0f ms), %u thread(s), memo %s\n", atlas.templates.size(), load_ms,
                build_ms, reader.threads(), memo ? "on" : "off");

    std::vector<win32::Frame> frames;
    for (const auto& s : bench::load_manifest(root + "/zenodo")) {
        if (frames.size() >= limit) break;
        win32::Frame f = win32::load_image(s.image);
        if (!f.empty()) frames.push_back(std::move(f));
    }

    uint64_t checksum = 1469598103934665603ull;
    auto mix = [&](const std::string& s) {
        for (unsigned char c : s) checksum = (checksum ^ c) * 1099511628211ull;
        checksum = (checksum ^ 0xff) * 1099511628211ull;
    };
    struct Acc {
        double layout = 0, ocr = 0, search = 0;
        uint64_t calls = 0, searches = 0, sads = 0, hits = 0;
    } cold, warm;
    std::map<std::string, std::pair<double, uint64_t>> by_kind;  // kind -> (ms, classify calls), cold only
    size_t boxes = 0, verified = 0;
    std::map<std::string, double> phases;
    LayoutParams lp;
    lp.profile = true;
    for (const auto& f : frames) {
        bench::Timer t;
        const Layout L = analyze_layout(f.view(), lp);
        for (const auto& [k, v] : L.timings) phases[k] += v;
        const double lay = t.ms();
        const Gray8 gray = to_luma(f.view());
        for (int pass = 0; pass < 2; ++pass) {
            Acc& a = pass ? warm : cold;
            const auto c0 = reader.counters();
            bench::Timer t2;
            const auto reads = reader.read_boxes(L, gray);
            a.ocr += t2.ms();
            a.layout += lay;
            const auto c1 = reader.counters();
            a.calls += c1.classify_calls - c0.classify_calls, a.searches += c1.search_calls - c0.search_calls;
            a.sads += c1.distance_evals - c0.distance_evals, a.hits += c1.memo_hits - c0.memo_hits;
            a.search += c1.search_ms - c0.search_ms;
            if (pass == 0) {
                boxes += reads.size();
                for (const auto& r : reads) verified += r.is_text, mix(r.text), mix(r.is_text ? "1" : "0");
            }
        }
        if (bench::has_flag(argc, argv, "--by-kind")) {  // separate cold run per kind, memo off
            ScreenReader solo(cls, {}, rp, vp, threads);
            solo.enable_memo(false);
            for (ElementKind k : {ElementKind::Text, ElementKind::Icon, ElementKind::Container, ElementKind::Image}) {
                Layout sub;
                for (const auto& e : L.elements)
                    if (e.kind == k) sub.elements.push_back(e);
                const auto c0 = solo.counters();
                bench::Timer t3;
                solo.read_boxes(sub, gray);
                by_kind[kind_name(k)].first += t3.ms();
                by_kind[kind_name(k)].second += solo.counters().classify_calls - c0.classify_calls;
            }
        }
    }
    const double n = double(frames.size());
    for (int pass = 0; pass < 2; ++pass) {
        const Acc& a = pass ? warm : cold;
        std::printf("%s: layout %5.1f ms | OCR %6.1f ms/frame | classify %6.0f calls, %6.0f searched (memo %4.1f%%), "
                    "search %6.1f ms, %5.1f us/search, %5.0f SAD/search\n",
                    pass ? "warm" : "cold", a.layout / n, a.ocr / n, double(a.calls) / n, double(a.searches) / n,
                    100.0 * double(a.hits) / double(std::max<uint64_t>(1, a.calls)), a.search / n,
                    1000.0 * a.search / double(std::max<uint64_t>(1, a.searches)), double(a.sads) / double(std::max<uint64_t>(1, a.searches)));
    }
    std::printf("%.0f boxes, %.0f verified text per frame\n", double(boxes) / n, double(verified) / n);
    std::printf("layout phases (ms/frame):");
    for (const auto& [k, v] : phases) std::printf(" %s %.1f |", k.c_str(), v / n);
    std::printf("\n");
    for (const auto& [k, v] : by_kind)
        std::printf("  by kind (cold, memo off) %-10s %7.1f ms/frame %7.0f classify calls/frame\n", k.c_str(), v.first / n, double(v.second) / n);
    std::printf("box cache hits: %llu\n", (unsigned long long)reader.box_hits());
    std::printf("reading checksum %016llx\n", (unsigned long long)checksum);
    return 0;
}

// === C:\Users\Cutie Magic 500\projects\creative\screen-ocr\bench\bench_segment.cpp ===
// Element bounding + text grouping + containment benchmark on Zenodo Desktop UI and WebUI.
//
//   bench_segment [--data datasets] [--overlays out/overlays] [--edge 28] [--word 0.55] [--line 1.25]
#include <cstdio>
#include <map>
#include <memory>
#include <set>

#include "common.hpp"
#include "dks/dks.hpp"
#include "dks/pipeline.hpp"

#include <fstream>

using namespace dks;

namespace {

// Map dataset labels onto our four element kinds (or "" = ignore for per-class scores).
std::string gt_group(const std::string& dataset, const std::string& label) {
    if (label == "Text") return "Text";
    if (label == "Icon" || label == "WebIcon") return "Icon";
    if (label == "Image") return "Image";
    if (dataset == "webui") return label == "generic" ? "" : "Container";
    return "Container";  // every widget / panel label in the Zenodo taxonomy
}

const ScreenReader* g_reader = nullptr;  // OCR verification pass (default on; --no-verify)
std::ofstream g_dump;                     // --ocr-dump: per-box OCR evidence

struct DatasetScores {
    std::map<std::string, eval::DetectionScore> det;  // per group + "All"
    eval::DetectionScore text_words, text_blocks;     // GT Text vs our word / paragraph boxes
    eval::GroupingScore group_lines, group_words, group_blocks;
    eval::DepthScore depth;
    double ms = 0;
    size_t images = 0, atoms = 0;
};

// DOM-derived GT (WebUI) boxes are element boxes (full-width headings, padded buttons), not ink.
// "Tight" mode shrinks each GT box to the extent of structure pixels inside it (the same polarity-free
// edge map for every method), and drops GT boxes with no visible structure.
std::vector<bench::GtBox> tighten(const std::vector<bench::GtBox>& gt, const win32::Frame& img, int edge_t) {
    Gray8 m(img.width, img.height);
    edge_mask(img.view(), m.view(), edge_t);
    std::vector<bench::GtBox> out;
    for (auto g : gt) {
        const Rect r = g.box.clip(img.width, img.height);
        int32_t l = INT32_MAX, t = INT32_MAX, rr = INT32_MIN, b = INT32_MIN;
        for (int32_t y = r.y; y < r.bottom(); ++y)
            for (int32_t x = r.x; x < r.right(); ++x)
                if (m.at(x, y)) l = std::min(l, x), rr = std::max(rr, x + 1), t = std::min(t, y), b = std::max(b, y + 1);
        if (l == INT32_MAX || rr - l < 2 || b - t < 3) continue;
        g.box = Rect::from_ltrb(l, t, rr, b);
        out.push_back(g);
    }
    return out;
}

void run(const std::string& root, const std::string& name, const LayoutParams& P, const std::string& overlay_dir,
         int max_overlays, bool tight) {
    const auto samples = bench::load_manifest(root + "/" + name);
    if (samples.empty()) { std::printf("  (no samples for %s)\n", name.c_str()); return; }
    DatasetScores S;
    int overlays = 0;
    for (const auto& s : samples) {
        win32::Frame img = win32::load_image(s.image);
        if (img.empty()) { std::printf("  ! cannot load %s\n", s.image.c_str()); continue; }
        auto gt = bench::load_gt(s.gt);
        if (tight) gt = tighten(gt, img, P.edge_threshold);

        bench::Timer t;
        Layout L;
        std::vector<ElementKind> kind;  // effective kind after OCR verification
        std::vector<uint8_t> drop;
        if (g_reader) {
            ScreenRead SR = g_reader->read(img.view());
            L = std::move(SR.layout);
            for (const auto& re : SR.elements) {
                ElementKind k = re.element.kind;
                bool d = false;
                if (k == ElementKind::Text && !re.is_text) d = true;             // unreadable "text": drop
                if (k == ElementKind::Icon && re.is_text) k = ElementKind::Text;  // readable "icon": promote
                kind.push_back(k);
                drop.push_back(d);
                if (g_dump) {
                    float best = 0;
                    for (const auto& g : gt)
                        if (g.label == "Text") best = std::max(best, re.element.bbox.iou(g.box));
                    const auto& r = re.detail;
                    float best_any = 0;
                    for (const auto& g : gt) best_any = std::max(best_any, re.element.bbox.iou(g.box));
                    g_dump << name << '\t' << kind_name(re.element.kind) << '\t' << re.element.bbox.w << '\t'
                           << re.element.bbox.h << '\t' << best << '\t' << r.glyphs.size() << '\t' << r.mean_dist << '\t'
                           << r.alnum_frac << '\t' << r.fit_residual << '\t' << r.line_fit << '\t' << re.text << '\t' << s.image << '\t'
                           << re.element.bbox.x << '\t' << re.element.bbox.y << '\t' << best_any << '\n';
                }
            }
        } else {
            L = analyze_layout(img.view(), P);
            for (const auto& e : L.elements) kind.push_back(e.kind), drop.push_back(0);
        }
        S.ms += t.ms();
        ++S.images;
        S.atoms += L.ccl.components.size();

        std::map<std::string, std::vector<Rect>> G, Pr;
        std::vector<Rect> gt_all, pred_all, words;
        for (const auto& g : gt) {
            const std::string grp = gt_group(name, g.label);
            if (!grp.empty()) G[grp].push_back(g.box);
            gt_all.push_back(g.box);
        }
        for (size_t i = 0; i < L.elements.size(); ++i) {
            if (drop[i]) continue;
            Pr[kind_name(kind[i])].push_back(L.elements[i].bbox);
            pred_all.push_back(L.elements[i].bbox);
        }
        // Words / blocks survive only where a kept text line covers them.
        auto covered = [&](const Rect& r) {
            for (const Rect& t : Pr["Text"])
                if (t.contains_point(r.x + r.w / 2, r.y + r.h / 2)) return true;
            return false;
        };
        for (const auto& w : L.words)
            if (!g_reader || covered(w.bbox)) words.push_back(w.bbox);
        std::vector<Rect> blocks;
        for (const auto& b : L.blocks)
            if (!g_reader || covered(b.bbox)) blocks.push_back(b.bbox);
        S.text_blocks.add(G["Text"], blocks);
        S.group_blocks.add(G["Text"], blocks);

        for (const char* k : {"Text", "Icon", "Image", "Container"}) S.det[k].add(G[k], Pr[k]);
        S.det["All"].add(gt_all, pred_all);
        S.text_words.add(G["Text"], words);
        S.group_lines.add(G["Text"], Pr["Text"]);
        S.group_words.add(G["Text"], words);
        S.depth.add(gt_all, pred_all);

        if (!overlay_dir.empty() && overlays < max_overlays) {
            win32::Frame o = img;
            for (const auto& g : gt) bench::draw_rect(o, g.box, 0, 200, 0);
            for (size_t i = 0; i < L.elements.size(); ++i) {
                if (drop[i]) continue;
                const Rect& bb = L.elements[i].bbox;
                switch (kind[i]) {
                    case ElementKind::Text: bench::draw_rect(o, bb, 255, 0, 0); break;
                    case ElementKind::Icon: bench::draw_rect(o, bb, 0, 90, 255); break;
                    case ElementKind::Image: bench::draw_rect(o, bb, 255, 0, 255); break;
                    case ElementKind::Container: bench::draw_rect(o, bb, 255, 160, 0); break;
                }
            }
            char path[512];
            std::snprintf(path, sizeof path, "%s/%s_%02d.png", overlay_dir.c_str(), name.c_str(), overlays++);
            win32::save_png(path, o);
        }
    }

    std::printf("\n== %s%s%s  (%zu images, %.1f ms/frame avg, %.0f CCL atoms/frame)\n", name.c_str(), tight ? " [ink-tight GT]" : "", g_reader ? " [layout+OCR verify]" : " [layout only]", S.images,
                S.ms / double(S.images), double(S.atoms) / double(S.images));
    std::printf("  %-22s %6s %6s %8s %8s %8s %8s %8s\n", "element bounding", "n_gt", "n_pred", "R@.50", "P@.50",
                "R@.75", "F1@.50", "mIoU");
    auto row = [](const char* k, const eval::DetectionScore& d) {
        std::printf("  %-22s %6zu %6zu %8.3f %8.3f %8.3f %8.3f %8.3f\n", k, d.n_gt, d.n_pred, d.recall50(),
                    d.precision50(), d.recall75(), d.f1_50(), d.mean_best_iou());
    };
    for (const char* k : {"Text", "Icon", "Image", "Container", "All"}) row(k, S.det[k]);
    row("Text (as words)", S.text_words);
    row("Text (as blocks)", S.text_blocks);
    std::printf("  %-22s %8s %8s %8s %8s %8s\n", "text grouping", "1:1", "split", "merge", "miss", "overlap");
    auto grow = [](const char* k, const eval::GroupingScore& g) {
        std::printf("  %-22s %8.3f %8.3f %8.3f %8.3f %8.3f\n", k, g.one_to_one(), g.split_rate(), g.merge_rate(),
                    g.miss_rate(), g.overlap_ratio());
    };
    grow("lines", S.group_lines);
    grow("words", S.group_words);
    grow("blocks", S.group_blocks);
    std::printf("  containment: matched %zu, depth precision %.3f, parent precision %.3f\n", S.depth.matched,
                S.depth.depth_precision(), S.depth.parent_precision());
}

}  // namespace

int main(int argc, char** argv) {
    const std::string root = bench::arg_value(argc, argv, "--data", "datasets");
    const std::string overlays = bench::arg_value(argc, argv, "--overlays", "");
    const int max_overlays = std::stoi(bench::arg_value(argc, argv, "--max-overlays", "4"));
    LayoutParams P;
    P.edge_threshold = std::stoi(bench::arg_value(argc, argv, "--edge", std::to_string(P.edge_threshold)));
    P.word_gap = std::stof(bench::arg_value(argc, argv, "--word", std::to_string(P.word_gap)));
    P.line_gap = std::stof(bench::arg_value(argc, argv, "--line", std::to_string(P.line_gap)));
    P.image_colors = std::stoi(bench::arg_value(argc, argv, "--img-colors", std::to_string(P.image_colors)));
    P.image_min_cells = std::stoi(bench::arg_value(argc, argv, "--img-cells", std::to_string(P.image_min_cells)));
    P.flat_min_fill = std::stof(bench::arg_value(argc, argv, "--flat-fill", std::to_string(P.flat_min_fill)));
    P.icon_colors = std::stoi(bench::arg_value(argc, argv, "--icon-colors", std::to_string(P.icon_colors)));
    P.gutter_veto = !bench::has_flag(argc, argv, "--no-gutter");
    P.luma_veto = !bench::has_flag(argc, argv, "--no-luma");
    P.weak_edge_threshold = std::stoi(bench::arg_value(argc, argv, "--weak", std::to_string(P.weak_edge_threshold)));
    P.luma_delta = std::stoi(bench::arg_value(argc, argv, "--luma-delta", std::to_string(P.luma_delta)));
    P.flat_max_std = std::stof(bench::arg_value(argc, argv, "--flat-std", std::to_string(P.flat_max_std)));
    P.image_min_edge_density = std::stof(bench::arg_value(argc, argv, "--img-edges", std::to_string(P.image_min_edge_density)));
    P.image_min_size = std::stoi(bench::arg_value(argc, argv, "--img-min", std::to_string(P.image_min_size)));
    P.image_min_fill = std::stof(bench::arg_value(argc, argv, "--img-fill", std::to_string(P.image_min_fill)));
    P.detect_images = !bench::has_flag(argc, argv, "--no-images");
    P.flat_containers = !bench::has_flag(argc, argv, "--no-flat");
    std::unique_ptr<ocr::Atlas> atlas;
    std::unique_ptr<ocr::Classifier> cls;
    std::unique_ptr<ScreenReader> reader;
    if (!bench::has_flag(argc, argv, "--no-verify")) {
        atlas = std::make_unique<ocr::Atlas>();
        std::string err;
        if (!atlas->load(bench::arg_value(argc, argv, "--atlas", "data/ui_fonts.dksa"), &err)) {
            std::fprintf(stderr, "atlas: %s (use --no-verify to skip OCR)\n", err.c_str());
            return 1;
        }
        atlas->prune(3.0f);
        cls = std::make_unique<ocr::Classifier>(*atlas, ocr::FeatureWeights{}, ocr::SearchMode::Sorted,
                                                std::stof(bench::arg_value(argc, argv, "--margin", "15")));
        cls->enable_memo(true);
        ocr::RecognizerParams rp;
        rp.topk = std::stoi(bench::arg_value(argc, argv, "--topk", std::to_string(rp.topk)));
        rp.atom_merge_gap = std::stof(bench::arg_value(argc, argv, "--atom-gap", std::to_string(rp.atom_merge_gap)));
    rp.atom_merge_px = std::stoi(bench::arg_value(argc, argv, "--atom-px", std::to_string(rp.atom_merge_px)));
        reader = std::make_unique<ScreenReader>(*cls, P, rp);
        g_reader = reader.get();
        const std::string dump = bench::arg_value(argc, argv, "--ocr-dump", "");
        if (!dump.empty()) g_dump.open(dump, std::ios::binary);
    }
    if (!overlays.empty()) CreateDirectoryA(overlays.c_str(), nullptr);
    const std::string only = bench::arg_value(argc, argv, "--only", "");
    if (!only.empty() && only != "zenodo" && only != "webui") {  // any other manifest dir, e.g. negatives
        run(root, only, P, overlays, max_overlays, false);
        return 0;
    }
    for (const char* ds : {"zenodo", "webui"})
        if (only.empty() || only == ds) {
            run(root, ds, P, overlays, max_overlays, false);
            if (std::string(ds) == "webui") run(root, ds, P, "", 0, true);
        }
    return 0;
}

// === C:\Users\Cutie Magic 500\projects\creative\screen-ocr\bench\ocr_debug.cpp ===
// Print the recogniser's glyph decisions for one crop.
//   ocr_debug <image> [x y w h] [--atlas data/ui_fonts.dksa]
#include <cstdio>

#include "common.hpp"
#include "dks/dks.hpp"
#include "dks/ocr/ocr.hpp"

using namespace dks;

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: ocr_debug <image> [x y w h]\n"); return 1; }
    ocr::Atlas atlas;
    std::string err;
    if (!atlas.load(bench::arg_value(argc, argv, "--atlas", "data/ui_fonts.dksa"), &err)) {
        std::fprintf(stderr, "%s\n", err.c_str());
        return 1;
    }
    atlas.prune(std::stof(bench::arg_value(argc, argv, "--prune", "3.0")));
    ocr::Classifier cls(atlas);
    ocr::Recognizer rec(cls);
    rec.params().trace = bench::has_flag(argc, argv, "--trace");
    win32::Frame img = win32::load_image(argv[1]);
    if (img.empty()) { std::fprintf(stderr, "cannot load\n"); return 1; }
    const Gray8 gray = to_luma(img.view());
    Rect r{0, 0, gray.width(), gray.height()};
    if (argc >= 6 && argv[2][0] != '-') r = Rect{std::atoi(argv[2]), std::atoi(argv[3]), std::atoi(argv[4]), std::atoi(argv[5])};
    const auto res = rec.recognize(gray.cview().sub(r));
    std::printf("text: \"%s\"  cost %.1f  baseline %.1f cap %.1f\n", res.utf8().c_str(), res.cost, res.baseline,
                res.cap_height);
    for (const auto& g : res.glyphs) {
        std::printf("  [%3d,%3d %2dx%2d] ", g.box.x, g.box.y, g.box.w, g.box.h);
        for (int c = 0; c < g.ncand; ++c) {
            std::string s;
            utf8_append(s, g.cand[c].ch);
            const auto& t = atlas.templates[g.cand[c].tmpl];
            std::printf("%s%s %.1f(%s %d) ", c == g.chosen ? "*" : "", s.c_str(), g.cand[c].dist,
                        atlas.fonts[t.font].c_str(), t.px);
        }
        std::printf("  eu%d la%.2f\n", g.feat.euler, g.feat.log_aspect);
        if (bench::has_flag(argc, argv, "--bmp"))
            for (int y = 0; y < 16; ++y) {
                std::printf("      ");
                for (int x = 0; x < 16; ++x) std::printf("%c", " .:-=+*#%@"[g.feat.bmp[size_t(y * 16 + x)] * 9 / 255]);
                std::printf("\n");
            }
    }
    return 0;
}

// === C:\Users\Cutie Magic 500\projects\creative\screen-ocr\bench\screen_ocr.cpp ===
// Live screen (or image) -> layout -> OCR pass over every box -> verified text.
//
//   screen_ocr                         capture the whole virtual desktop once
//   screen_ocr --frames 5              capture 5 frames (shows the memo cache warming up)
//   screen_ocr --image shot.png        run on a file instead of the screen
//   screen_ocr --out overlay.png       debug overlay (red = verified text, grey = rejected text,
//                                      blue = icon, orange = container, magenta = image)
//   screen_ocr --tsv boxes.tsv         every box: x y w h kind depth is_text confidence text
//   screen_ocr --save-frame raw.png   also save the raw captured frame
//   screen_ocr --threads 1             force single-threaded OCR
//   screen_ocr --no-verify             keep every layout Text box (no OCR verification)
#include <cstdio>
#include <fstream>

#include "common.hpp"
#include "dks/dks.hpp"
#include "dks/pipeline.hpp"

using namespace dks;

int main(int argc, char** argv) {
    SetProcessDPIAware();  // physical pixels on scaled displays
    const std::string image = bench::arg_value(argc, argv, "--image", "");
    const std::string out_png = bench::arg_value(argc, argv, "--out", "");
    const std::string out_tsv = bench::arg_value(argc, argv, "--tsv", "");
    const std::string save_frame = bench::arg_value(argc, argv, "--save-frame", "");
    const int frames = std::stoi(bench::arg_value(argc, argv, "--frames", "1"));
    const unsigned threads = unsigned(std::stoul(bench::arg_value(argc, argv, "--threads", "0")));
    const bool quiet = bench::has_flag(argc, argv, "--quiet");

    bench::Timer tl;
    ocr::Atlas atlas;
    std::string err;
    if (!atlas.load(bench::arg_value(argc, argv, "--atlas", "data/ui_fonts.dksa"), &err)) {
        std::fprintf(stderr, "atlas: %s\n", err.c_str());
        return 1;
    }
    atlas.prune(3.0f);
    const ocr::Classifier cls(atlas);
    TextVerifyParams vp;
    vp.enabled = !bench::has_flag(argc, argv, "--no-verify");
    ScreenReader reader(cls, {}, {}, vp, threads);
    reader.enable_memo(true);
    std::printf("atlas ready: %zu templates (%.0f ms), %u OCR threads\n", atlas.templates.size(), tl.ms(), reader.threads());

    for (int f = 0; f < frames; ++f) {
        bench::Timer tc;
        win32::Frame frame = image.empty() ? win32::capture_screen() : win32::load_image(image);
        if (frame.empty()) { std::fprintf(stderr, "no frame\n"); return 1; }
        const double cap_ms = tc.ms();
        if (!save_frame.empty() && f == 0) win32::save_png(save_frame, frame);

        bench::Timer tr;
        const ScreenRead SR = reader.read(frame.view());
        const double read_ms = tr.ms();
        size_t n_text = 0, n_rejected = 0, n_chars = 0;
        for (const auto& e : SR.elements) {
            if (e.is_text) ++n_text, n_chars += e.text.size();
            else if (e.element.kind == ElementKind::Text) ++n_rejected;
        }
        std::printf("frame %d: %dx%d  capture %.1f ms | layout+OCR %.1f ms | %zu boxes, %zu verified text (%zu chars), "
                    "%zu layout-text rejected by OCR\n",
                    f, frame.width, frame.height, cap_ms, read_ms, SR.elements.size(), n_text, n_chars, n_rejected);

        if (f + 1 < frames) continue;
        if (!quiet)
            for (const auto& e : SR.elements)
                if (e.is_text)
                    std::printf("  [%4d,%4d %4dx%3d] %.2f %s\n", e.element.bbox.x, e.element.bbox.y, e.element.bbox.w,
                                e.element.bbox.h, e.confidence, e.text.c_str());
        if (!out_tsv.empty()) {
            std::ofstream o(out_tsv, std::ios::binary);
            for (const auto& e : SR.elements) {
                std::string t = e.text;
                for (char& c : t)
                    if (c == '\t' || c == '\n') c = ' ';
                o << e.element.bbox.x << '\t' << e.element.bbox.y << '\t' << e.element.bbox.w << '\t' << e.element.bbox.h << '\t'
                  << kind_name(e.element.kind) << '\t' << e.element.depth << '\t' << int(e.is_text) << '\t' << e.confidence
                  << '\t' << t << '\n';
            }
        }
        if (!out_png.empty()) {
            win32::Frame o = frame;
            for (const auto& e : SR.elements) {
                const Rect& b = e.element.bbox;
                if (e.is_text) { bench::draw_rect(o, b, 255, 0, 0); continue; }
                switch (e.element.kind) {
                    case ElementKind::Text: bench::draw_rect(o, b, 150, 150, 150); break;
                    case ElementKind::Icon: bench::draw_rect(o, b, 0, 90, 255); break;
                    case ElementKind::Image: bench::draw_rect(o, b, 255, 0, 255); break;
                    case ElementKind::Container: bench::draw_rect(o, b, 255, 160, 0); break;
                }
            }
            win32::save_png(out_png, o);
        }
    }
    return 0;
}

// === C:\Users\Cutie Magic 500\projects\creative\screen-ocr\bench\screen_pipeline.cpp ===
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

// === C:\Users\Cutie Magic 500\projects\creative\screen-ocr\bench\train_pack.cpp ===
// Compile a labelled crop dataset into a detection pack (.dkpk).
//
//   train_pack --classes DIR  --name NAME --tag-key KEY --out data/packs/NAME.dkpk [--k 3] [--holdout 5]
//       DIR/<label>/*.png|jpg|bmp|gif|webp     one sub-folder per class, folder name = label
//   train_pack --manifest DIR ...
//       DIR/manifest.tsv + *.gt.tsv             the repo's dataset format; label = column 5 of the first row
//
// --holdout N   evaluate first: every N-th crop of each class (deterministic) is held out, accuracy and a
//               confusion summary are printed, then the pack is trained on *all* crops and saved.
// See docs/PACKS.md.
#include <algorithm>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

#include "common.hpp"
#include "dks/detection/detection_pack.hpp"

using namespace dks;

namespace {

struct Item {
    std::string label, path;
    Gray8 luma;
};

std::vector<std::string> list_dir(const std::string& dir, bool dirs) {
    std::vector<std::string> out;
    WIN32_FIND_DATAA fd;
    HANDLE h = FindFirstFileA((dir + "/*").c_str(), &fd);
    for (BOOL ok = h != INVALID_HANDLE_VALUE; ok; ok = FindNextFileA(h, &fd)) {
        const std::string n = fd.cFileName;
        if (n == "." || n == "..") continue;
        const bool is_dir = (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0;
        if (is_dir == dirs) out.push_back(n);
    }
    if (h != INVALID_HANDLE_VALUE) FindClose(h);
    std::sort(out.begin(), out.end());  // deterministic order
    return out;
}

bool is_image(const std::string& n) {
    std::string e = n.substr(n.find_last_of('.') + 1);
    std::transform(e.begin(), e.end(), e.begin(), [](unsigned char c) { return char(std::tolower(c)); });
    return e == "png" || e == "jpg" || e == "jpeg" || e == "bmp" || e == "gif" || e == "webp" || e == "tif" || e == "tiff";
}

}  // namespace

int main(int argc, char** argv) {
    const std::string classes = bench::arg_value(argc, argv, "--classes", "");
    const std::string manifest = bench::arg_value(argc, argv, "--manifest", "");
    const std::string name = bench::arg_value(argc, argv, "--name", "my_pack");
    const std::string key = bench::arg_value(argc, argv, "--tag-key", "widget");
    const std::string out = bench::arg_value(argc, argv, "--out", "data/packs/" + name + ".dkpk");
    const int k = std::stoi(bench::arg_value(argc, argv, "--k", "3"));
    const int holdout = std::stoi(bench::arg_value(argc, argv, "--holdout", "0"));
    if (classes.empty() == manifest.empty()) {
        std::fprintf(stderr, "usage: train_pack (--classes DIR | --manifest DIR) --name NAME --tag-key KEY [--out FILE] [--k 3] [--holdout 5]\n");
        return 2;
    }

    std::vector<Item> items;
    auto add = [&](const std::string& label, const std::string& path) {
        win32::Frame f = win32::load_image(path);
        if (f.empty()) { std::fprintf(stderr, "  ! cannot read %s\n", path.c_str()); return; }
        items.push_back({label, path, to_luma(f.view())});
    };
    if (!classes.empty()) {
        for (const auto& label : list_dir(classes, true))
            for (const auto& file : list_dir(classes + "/" + label, false))
                if (is_image(file)) add(label, classes + "/" + label + "/" + file);
    } else {
        for (const auto& s : bench::load_manifest(manifest)) {
            const auto gt = bench::load_gt(s.gt);
            if (!gt.empty()) add(gt[0].label, s.image);
        }
    }
    std::map<std::string, int> per_class;
    for (const auto& it : items) ++per_class[it.label];
    std::printf("%zu crops, %zu classes\n", items.size(), per_class.size());
    for (const auto& [l, c] : per_class) std::printf("  %-28s %5d%s\n", l.c_str(), c, c < 5 ? "   (few examples: expect low recall)" : "");
    if (items.empty()) return 1;

    if (holdout > 1) {
        detect::TemplatePack eval(name, key, {}, k);
        std::map<std::string, int> seen;
        std::vector<const Item*> test;
        for (const auto& it : items)
            if (++seen[it.label] % holdout == 0) test.push_back(&it);
            else eval.add_example(it.label, it.luma.cview());
        std::map<std::string, std::map<std::string, int>> conf;
        int ok = 0;
        for (const Item* it : test) {
            const auto m = eval.classify(detect::describe(it->luma.cview()));
            const std::string pred = m.empty() ? "?" : m.front().tag;
            ++conf[it->label][pred];
            ok += pred == it->label;
        }
        std::printf("\nhold-out (every %d-th crop per class): accuracy %.3f (%d / %zu)\n", holdout,
                    test.empty() ? 0.0 : double(ok) / double(test.size()), ok, test.size());
        for (const auto& [gt, row] : conf) {
            int tot = 0, hit = 0;
            std::string worst;
            int worst_n = 0;
            for (const auto& [p, c] : row) {
                tot += c, hit += p == gt ? c : 0;
                if (p != gt && c > worst_n) worst = p, worst_n = c;
            }
            std::printf("  %-28s recall %.2f  (%d/%d)%s%s\n", gt.c_str(), double(hit) / double(tot), hit, tot,
                        worst.empty() ? "" : "   most confused with ", worst.c_str());
        }
    }

    detect::TemplatePack pack(name, key, {}, k);
    for (const auto& it : items) pack.add_example(it.label, it.luma.cview());
    if (!pack.save(out)) { std::fprintf(stderr, "cannot write %s\n", out.c_str()); return 1; }
    std::printf("\nwrote %s  (%zu exemplars, %zu classes, k=%d, tag key '%s')\n", out.c_str(), pack.size(), pack.classes().size(), k,
                key.c_str());
    return 0;
}

// === C:\Users\Cutie Magic 500\projects\creative\screen-ocr\examples\pipeline_example.cpp ===
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

// === C:\Users\Cutie Magic 500\projects\creative\screen-ocr\tests\test_core.cpp ===
// Minimal self-checking tests for the deterministic primitives (no framework needed).
#include <cstdio>
#include <string>

#include "dks/dks.hpp"
#include "dks/ocr/features.hpp"

using namespace dks;

static int failures = 0;
#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) { std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); ++failures; } \
    } while (0)

static Gray8 from_ascii(const char* const* rows, int h) {
    const int w = int(std::string(rows[0]).size());
    Gray8 g(w, h, 0);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) g.at(x, y) = rows[y][x] == '#' ? 1 : 0;
    return g;
}

int main() {
    // --- geometry
    const Rect a{0, 0, 10, 10}, b{5, 5, 10, 10};
    CHECK(a.inter_area(b) == 25);
    CHECK(std::abs(a.iou(b) - 25.f / 175.f) < 1e-6f);
    CHECK(Rect({0, 0, 100, 100}).contains(Rect{10, 10, 5, 5}));
    CHECK(!a.contains(b));

    // --- CCL: 8- vs 4-connectivity, diagonal touch
    const char* diag[] = {"#..", ".#.", "..#"};
    const Gray8 d = from_ascii(diag, 3);
    CHECK(label_components(d.cview(), Connectivity::Eight).components.size() == 1);
    CHECK(label_components(d.cview(), Connectivity::Four).components.size() == 3);

    // U shape: two arms joined only at the bottom row must merge (union across rows)
    const char* u[] = {"#...#", "#...#", "#####"};
    const LabelResult lu = label_components(from_ascii(u, 3).cview());
    CHECK(lu.components.size() == 1);
    CHECK(lu.components[0].area == 9);
    CHECK((lu.components[0].bbox == Rect{0, 0, 5, 3}));

    // --- Euler number / holes / components
    const char* ring[] = {"#####", "#...#", "#...#", "#####"};             // 'o': 1 component, 1 hole
    const char* eight[] = {"###", "#.#", "###", "#.#", "###"};           // 'B'/'8': 2 holes
    const char* colon[] = {"#", ".", ".", "#"};                          // ':' : 2 components
    CHECK(ocr::euler_number(from_ascii(ring, 4).cview()) == 0);
    CHECK(ocr::euler_number(from_ascii(eight, 5).cview()) == -1);
    CHECK(ocr::euler_number(from_ascii(colon, 4).cview()) == 2);
    CHECK(ocr::euler_number(d.cview()) == 1);  // diagonal line is one 8-connected component

    // --- containment forest
    const std::vector<Rect> boxes = {{0, 0, 100, 100}, {10, 10, 50, 50}, {20, 20, 5, 5}, {200, 0, 10, 10}};
    const Forest f = build_containment_forest(boxes);
    CHECK(f.parent[0] == -1 && f.parent[1] == 0 && f.parent[2] == 1 && f.parent[3] == -1);
    CHECK(f.depth[2] == 2);

    // --- metrics
    CHECK(eval::levenshtein(U"kitten", U"sitting") == 3);
    eval::TextScore ts;
    ts.add("Hello", "hello");
    CHECK(ts.edits == 1 && ts.edits_ci == 0 && ts.exact_ci == 1);
    CHECK(utf8_encode(utf8_decode("€£©")) == "€£©");

    // --- feature determinism: identical input -> identical descriptor
    const Gray8 r = from_ascii(ring, 4);
    Gray8 ink(r.width(), r.height());
    for (int y = 0; y < r.height(); ++y)
        for (int x = 0; x < r.width(); ++x) ink.at(x, y) = r.at(x, y) ? 255 : 0;
    const auto f1 = ocr::make_feature(ink.cview(), r.cview()), f2 = ocr::make_feature(ink.cview(), r.cview());
    CHECK(f1.bmp == f2.bmp && f1.holes == 1 && f1.ncomp == 1);
    CHECK(ocr::distance(f1, f2) == 0.f);
    CHECK(ocr::lower_bound(f1, f2) <= ocr::distance(f1, f2));

    std::printf(failures ? "%d FAILURES\n" : "all tests passed\n", failures);
    return failures ? 1 : 0;
}

// === C:\Users\Cutie Magic 500\projects\creative\screen-ocr\tests\test_pipeline.cpp ===
// Tests for the modular layer: LaTeX spatial parser, LaTeX canonicalisation, pack registry +
// DetectionStage chaining, Pipeline stage management, and (when the atlas is present) equivalence of
// the pipeline's layout+text stages with ScreenReader::read(). No downloaded data needed.
#include <cstdio>
#include <fstream>
#include <string>

#include "dks/detection/detection_pack.hpp"
#include "dks/eval/latex_metrics.hpp"
#include "dks/latex/spatial_tree_parsing.hpp"
#include "dks/pipeline/core_stages.hpp"

using namespace dks;

static int failures = 0;
#define CHECK(cond)                                                                                  \
    do {                                                                                             \
        if (!(cond)) { std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); ++failures; } \
    } while (0)
#define CHECK_EQ(a, b)                                                                                              \
    do {                                                                                                            \
        const std::string _a = (a), _b = (b);                                                                       \
        if (_a != _b) { std::printf("FAIL %s:%d\n   got      %s\n   expected %s\n", __FILE__, __LINE__, _a.c_str(), _b.c_str()); ++failures; } \
    } while (0)

// Synthetic symbol: box, label; scale = cap height, baseline = y of the glyph's baseline.
static latex::MathSymbol sym(int x, int y, int w, int h, char32_t ch, float scale, float baseline) {
    latex::MathSymbol s;
    s.box = Rect{x, y, w, h};
    s.ch = ch;
    s.tex = latex::tex_of(ch);
    s.scale = scale;
    s.baseline = baseline;
    return s;
}
static latex::MathSymbol bar(int x, int y, int w) {
    latex::MathSymbol s;
    s.kind = latex::MathSymbol::Kind::Bar;
    s.box = Rect{x, y, w, 1};
    s.ch = U'-';
    s.tex = "-";
    s.scale = float(w) / 1.14f;
    s.baseline = float(y) + 0.5f + 0.37f * s.scale;
    return s;
}

static void test_parser() {
    // x^2 + y_i = 1   (main size 10, baseline y=20)
    std::vector<latex::MathSymbol> v = {
        sym(0, 13, 6, 7, U'x', 10, 20),  sym(7, 8, 4, 6, U'2', 6, 14),     sym(14, 12, 7, 7, U'+', 10, 20),
        sym(24, 13, 6, 9, U'y', 10, 20), sym(31, 19, 3, 5, U'i', 6, 24),   sym(38, 14, 7, 4, U'=', 10, 20),
        sym(48, 10, 4, 10, U'1', 10, 20)};
    CHECK_EQ(latex::parse_latex(v), "x ^ { 2 } + y _ { i } = 1");

    // \frac { a } { b }
    v = {sym(2, 2, 6, 7, U'a', 10, 9), bar(0, 11, 10), sym(2, 13, 6, 10, U'b', 10, 23)};
    CHECK_EQ(latex::parse_latex(v), "\\frac { a } { b }");

    // minus between two symbols stays a minus
    v = {sym(0, 13, 6, 7, U'a', 10, 20), bar(8, 16, 8), sym(18, 10, 6, 10, U'b', 10, 20)};
    CHECK_EQ(latex::parse_latex(v), "a - b");

    // \sqrt { x }
    latex::MathSymbol r;
    r.kind = latex::MathSymbol::Kind::Radical;
    r.box = Rect{0, 5, 16, 16};
    r.radicand_x = 6;
    r.tex = "\\sqrt";
    v = {r, sym(8, 12, 6, 7, U'x', 10, 19)};
    CHECK_EQ(latex::parse_latex(v), "\\sqrt { x }");

    // \sum _ { i } ^ { n }
    v = {sym(0, 10, 14, 14, 0x2211, 14, 24), sym(4, 0, 5, 6, U'n', 6, 6), sym(5, 26, 3, 6, U'i', 6, 32)};
    CHECK_EQ(latex::parse_latex(v), "\\sum _ { i } ^ { n }");

    // a lone one-like stem in a subscript reads as 1
    v = {sym(0, 13, 6, 7, U'x', 10, 20), sym(7, 19, 3, 6, U'l', 6, 24)};
    CHECK_EQ(latex::parse_latex(v), "x _ { 1 }");
}

static void test_latex_canonical() {
    eval::LatexScore s;
    s.add("\\left( x \\right) ^ { 2 } \\, \\le { y }", "( x ) ^ 2 \\leq y");
    CHECK(s.exact == 1);
    eval::LatexScore t;
    t.add("\\frac { 1 } { 2 }", "\\frac { 1 } { 3 }");
    CHECK(t.exact == 0 && t.edits == 1);
}

// Dummy packs to exercise registry order and tag chaining.
struct TypePack : detect::IShapeClassifier {
    std::string name() const override { return "type"; }
    std::string tag_key() const override { return "widget"; }
    bool applies(const detect::Patch&) const override { return true; }
    std::vector<detect::ShapeMatch> classify(const detect::Patch& p) const override {
        return {{p.box.w > p.box.h ? "ui.switch" : "ui.checkbox", 0.9f, 0}};
    }
};
struct StatePack : detect::IShapeClassifier {
    std::string name() const override { return "state"; }
    std::string tag_key() const override { return "widget.state"; }
    bool applies(const detect::Patch& p) const override {
        const auto* t = p.tag("widget");
        return t && t->value == "ui.checkbox";
    }
    std::vector<detect::ShapeMatch> classify(const detect::Patch&) const override { return {{"checked", 0.8f, 0}}; }
};

static void test_packs_and_pipeline() {
    // Frame: white page with a square box and a wide box (both hollow outlines).
    const int W = 120, H = 60;
    std::vector<uint8_t> px(size_t(W) * H * 4, 255);
    auto rect = [&](int x0, int y0, int w, int h) {
        for (int y = y0; y < y0 + h; ++y)
            for (int x = x0; x < x0 + w; ++x)
                if (y == y0 || y == y0 + h - 1 || x == x0 || x == x0 + w - 1)
                    for (int c = 0; c < 3; ++c) px[(size_t(y) * W + x) * 4 + c] = 0;
    };
    rect(10, 10, 30, 30);
    rect(60, 20, 50, 20);
    const ColorView frame{px.data(), W, H, W * 4, PixelFormat::BGRA32};

    auto reg = std::make_shared<detect::PackRegistry>();
    reg->add(std::make_shared<TypePack>());
    reg->add(std::make_shared<StatePack>());
    CHECK(reg->packs().size() == 2);
    reg->add(std::make_shared<StatePack>());  // same name replaces
    CHECK(reg->packs().size() == 2);

    Pipeline pipe;
    pipe.add(std::make_shared<LayoutStage>()).add(std::make_shared<detect::DetectionStage>(reg, 2, 0.5f));
    CHECK(pipe.names().size() == 2);
    AnalysisContext ctx = pipe.run(frame);
    CHECK(ctx.has_layout && !ctx.layout.elements.empty());
    int checkbox = 0, checked = 0, sw = 0;
    for (size_t i = 0; i < ctx.layout.elements.size(); ++i) {
        const auto* t = ctx.find_tag(i, "widget");
        const auto* s = ctx.find_tag(i, "widget.state");
        if (t && t->value == "ui.checkbox") ++checkbox, checked += s != nullptr;
        if (t && t->value == "ui.switch") {
            ++sw;
            CHECK(s == nullptr);  // state pack must not run on switches
        }
    }
    CHECK(checkbox >= 1 && checked == checkbox && sw >= 1);

    // Removing a pack at runtime removes its tags on the next run; disabling a stage skips it.
    reg->remove("state");
    ctx = pipe.run(frame);
    for (size_t i = 0; i < ctx.layout.elements.size(); ++i) CHECK(ctx.find_tag(i, "widget.state") == nullptr);
    pipe.set_enabled("detection", false);
    ctx = pipe.run(frame);
    for (size_t i = 0; i < ctx.layout.elements.size(); ++i) CHECK(ctx.find_tag(i, "widget") == nullptr);
    CHECK(pipe.remove("detection") && pipe.names().size() == 1);

    // Cursor masking inpaints the rect and records an exclusion.
    Pipeline p2;
    p2.add(std::make_shared<CursorMaskStage>([](Rect& r) { r = Rect{5, 5, 8, 8}; return true; }));
    const AnalysisContext c2 = p2.run(frame);
    CHECK(c2.exclusions.size() == 1 && !c2.owned.empty());
    CHECK(px[(size_t(10) * W + 10) * 4] == 0);  // original frame untouched (copy-on-write)
}

static void test_equivalence() {
    ocr::Atlas atlas;
    if (!atlas.load("data/ui_fonts.dksa")) {
        std::printf("  (skip equivalence: data/ui_fonts.dksa not found)\n");
        return;
    }
    atlas.prune(3.0f);
    const ocr::Classifier cls(atlas);
    auto reader = std::make_shared<ScreenReader>(cls);
    // Synthetic frame with bars that look like text lines.
    const int W = 200, H = 60;
    std::vector<uint8_t> px(size_t(W) * H * 4, 255);
    for (int k = 0; k < 12; ++k)
        for (int y = 20; y < 30; ++y)
            for (int x = 10 + k * 14; x < 10 + k * 14 + 3 + (k % 3); ++x)
                for (int c = 0; c < 3; ++c) px[(size_t(y) * W + x) * 4 + c] = 20;
    const ColorView frame{px.data(), W, H, W * 4, PixelFormat::BGRA32};
    const ScreenRead direct = reader->read(frame);
    Pipeline pipe;
    pipe.add(std::make_shared<LayoutStage>()).add(std::make_shared<TextReadStage>(reader));
    const AnalysisContext ctx = pipe.run(frame);
    CHECK(ctx.reads.size() == direct.elements.size());
    for (size_t i = 0; i < ctx.reads.size() && i < direct.elements.size(); ++i) {
        CHECK(ctx.reads[i].text == direct.elements[i].text);
        CHECK(ctx.reads[i].is_text == direct.elements[i].is_text);
        CHECK(ctx.reads[i].element.bbox == direct.elements[i].element.bbox);
    }
}

int main() {
    test_parser();
    test_latex_canonical();
    test_packs_and_pipeline();
    test_equivalence();
    std::printf(failures ? "%d FAILURES\n" : "all pipeline tests passed\n", failures);
    return failures ? 1 : 0;
}

