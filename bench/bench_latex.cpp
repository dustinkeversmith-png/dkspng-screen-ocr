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
