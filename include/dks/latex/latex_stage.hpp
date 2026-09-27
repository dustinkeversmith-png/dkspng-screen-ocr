#pragma once
// LatexStage: the LaTeX character + structure stage as a separable pipeline step.
//
// It is independent of the base text stage: it has its own atlas (math_fonts.dksa), its own
// segmentation (no shared baseline) and its own parser. It considers a box only when
//   * the box is a text candidate (layout Text) that the text stage did not verify as one-baseline
//     text, or `claim_all_text` is set (pages known to be math); and
//   * the math evidence passes (see MathEvidence / LatexStageParams): enough symbols, good template
//     matches, few rare symbols, and — unless claim_all_text — genuine math content.
// Claimed boxes get {"latex", "<tokens>"}; nothing else about the element changes.
#include <memory>
#include <numeric>
#include <string>

#include "../pipeline/pipeline_stage.hpp"
#include "spatial_tree_parsing.hpp"

namespace dks::latex {

struct LatexStageParams {
    bool claim_all_text = false;  // math-only pages: read every Text box as math
    // Gating chosen on im2latex (positives) vs Zenodo desktops + stock wallpapers (negatives), see
    // bench_latex_stage: ~27% of formula images claimed at <= 0.1 false claims per screen.
    int min_symbols = 8;
    float max_mean_dist = 20.f;   // mean template distance of the glyph symbols
    float max_rare = 0.1f;        // share of symbols outside the common math vocabulary
    int min_math_marks = 3;       // structure (^ _ \frac \sqrt) + math-only symbols (Greek, relations, big ops)
    int max_height = 400;
    // Formula regions: the layout splits a formula at scripts / fraction parts (its line rules want one
    // baseline), so text boxes are grown by these fractions of their height and overlapping ones merged.
    float grow_x = 0.8f, grow_y = 0.5f;
    int max_region_members = 24;
};

// Evidence that a crop is a formula.
struct MathEvidence {
    int symbols = 0;
    float mean_dist = 0;
    float rare = 0;      // share of glyphs with a prior penalty (rare in math) or outside the vocabulary
    int math_marks = 0;  // scripts / fractions / radicals + math-only symbols (Greek, operators, relations)
    std::string tex;
};

inline MathEvidence math_evidence(const MathReader& reader, GrayView crop, const ParseParams& pp = {}) {
    MathEvidence ev;
    const auto syms = reader.symbols(crop);
    ev.symbols = int(syms.size());
    if (syms.empty()) return ev;
    float d = 0;
    int n = 0, rare = 0;
    for (const auto& s : syms) {
        if (s.kind == MathSymbol::Kind::Radical) ++ev.math_marks;
        if (s.kind != MathSymbol::Kind::Glyph) continue;
        d += s.dist, ++n;
        if (MathReader::prior(s.ch) >= 5.f) ++rare;
        const SymClass k = class_of(s.ch);
        if ((s.ch >= 0x0391 && s.ch <= 0x03F5) || k == SymClass::BigOp || k == SymClass::Relation) ++ev.math_marks;
    }
    ev.mean_dist = n ? d / float(n) : 99.f;
    ev.rare = n ? float(rare) / float(n) : 1.f;
    ev.tex = parse_latex(syms, pp);
    for (const char* m : {"^ {", "_ {", "\\frac", "\\sqrt"})
        for (size_t p = ev.tex.find(m); p != std::string::npos; p = ev.tex.find(m, p + 1)) ++ev.math_marks;
    return ev;
}

class LatexStage : public IPipelineStage {
public:
    LatexStage(std::shared_ptr<const ocr::Classifier> math_classifier, LatexStageParams p = {}, MathReadParams mp = {},
               ParseParams pp = {}, MathPrior prior = {})
        : cls_(std::move(math_classifier)), reader_(*cls_, mp, std::move(prior)), p_(p), pp_(pp) {}
    const char* name() const override { return "latex"; }

    // Candidate test on the text stage's reading of the box (if it ran).
    bool candidate(const AnalysisContext& ctx, size_t i) const {
        const Element& e = ctx.layout.elements[i];
        if (e.kind != ElementKind::Text || e.bbox.h > p_.max_height) return false;
        if (p_.claim_all_text || i >= ctx.reads.size()) return true;
        const auto& r = ctx.reads[i];
        return !(r.is_text && r.detail.line_fit && r.detail.fit_residual < 0.25f);  // one-baseline text: not ours
    }

    bool accept(const MathEvidence& ev) const {
        if (p_.claim_all_text) return ev.symbols >= 2;  // caller asserts the content is math
        if (ev.symbols < p_.min_symbols || ev.mean_dist > p_.max_mean_dist || ev.rare > p_.max_rare) return false;
        return ev.math_marks >= p_.min_math_marks;
    }

    // Candidate formula regions: grown text boxes merged by overlap. A region is a candidate when at
    // least one member is a candidate (not verified one-baseline text).
    std::vector<Region> regions(const AnalysisContext& ctx) const {
        const auto& E = ctx.layout.elements;
        std::vector<uint32_t> ids;
        std::vector<Rect> grown;
        for (uint32_t i = 0; i < E.size(); ++i) {
            if (E[i].kind != ElementKind::Text || E[i].bbox.h > p_.max_height) continue;
            const Rect& b = E[i].bbox;
            const int32_t gx = int32_t(p_.grow_x * float(b.h)), gy = int32_t(p_.grow_y * float(b.h));
            ids.push_back(i);
            grown.push_back(Rect{b.x - gx, b.y - gy, b.w + 2 * gx, b.h + 2 * gy});
        }
        if (ids.empty()) return {};
        std::vector<uint32_t> par(ids.size());
        std::iota(par.begin(), par.end(), 0u);
        auto find = [&](uint32_t i) { while (par[i] != i) i = par[i] = par[par[i]]; return i; };
        for (size_t a = 0; a < ids.size(); ++a)
            for (size_t b = a + 1; b < ids.size(); ++b)
                if (grown[a].inter_area(grown[b]) > 0) {
                    const uint32_t x = find(uint32_t(a)), y = find(uint32_t(b));
                    if (x != y) par[std::max(x, y)] = std::min(x, y);
                }
        std::vector<Region> out;
        std::vector<int32_t> slot(ids.size(), -1);
        for (size_t a = 0; a < ids.size(); ++a) {
            const uint32_t r = find(uint32_t(a));
            if (slot[r] < 0) slot[r] = int32_t(out.size()), out.emplace_back();
            Region& g = out[size_t(slot[r])];
            g.members.push_back(ids[a]);
            g.box = g.box.unite(E[ids[a]].bbox);
        }
        std::vector<Region> keep;
        for (auto& g : out) {
            if (int(g.members.size()) > p_.max_region_members) continue;  // a whole text column, not a formula
            bool any = false;
            for (uint32_t m : g.members) any |= candidate(ctx, m);
            if (!any) continue;
            // include scripts / limits that sit just outside the member boxes
            int32_t h = 0;
            for (uint32_t m : g.members) h = std::max(h, E[m].bbox.h);
            g.box = g.box.inflate(std::max<int32_t>(2, h * 3 / 10)).clip(ctx.frame.width, ctx.frame.height);
            keep.push_back(std::move(g));
        }
        return keep;
    }

    void process(AnalysisContext& ctx) const override {
        if (!ctx.has_layout) return;
        const Gray8& luma = ctx.ensure_luma();
        for (Region& g : regions(ctx)) {
            const MathEvidence ev = math_evidence(reader_, luma.cview().sub(g.box), pp_);
            if (!accept(ev)) continue;
            const float score = std::clamp(1.f - ev.mean_dist / (2.f * p_.max_mean_dist), 0.f, 1.f);
            g.tags.push_back(SemanticTag{"latex", ev.tex, score, name()});
            for (uint32_t m : g.members) ctx.tag(m, "latex.region", std::to_string(ctx.regions.size()), score, name());
            ctx.regions.push_back(std::move(g));
        }
    }

    const MathReader& reader() const noexcept { return reader_; }
    LatexStageParams& params() noexcept { return p_; }

private:
    std::shared_ptr<const ocr::Classifier> cls_;
    MathReader reader_;
    LatexStageParams p_;
    ParseParams pp_;
};

}  // namespace dks::latex
