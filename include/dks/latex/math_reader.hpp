#pragma once
// MathReader: symbol-level reading of a formula crop (Layer 2 of the LaTeX stage).
//
// Unlike the text recogniser there is no shared-baseline assumption: every symbol is found on its own
// (connected components, with stacked parts merged: i j = ≤ ≥ ≡ : ; ! ÷), classified against the
// math atlas, and given a *scale* (cap height implied by its best template) and a *baseline*. Two
// structural symbols are found without templates:
//   Bar      horizontal rule: minus / fraction bar / overline (decided by the parser from context)
//   Radical  √ with its vinculum: a hook on the left plus a bar running over the radicand
#include <algorithm>
#include <cmath>
#include <numeric>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "../ocr/ocr.hpp"
#include "math_symbols.hpp"

namespace dks::latex {

struct MathSymbol {
    enum class Kind : uint8_t { Glyph, Bar, Radical };
    Kind kind = Kind::Glyph;
    Rect box;
    char32_t ch = 0;
    std::string tex;
    float dist = 0;              // template distance (Glyph)
    float scale = 0;             // implied cap height in px (0 = unknown)
    float baseline = 0;          // implied baseline y in px
    int32_t radicand_x = 0;      // Radical: x where the vinculum starts
    ocr::Candidate cand[4];
    int ncand = 0;
};

struct MathReadParams {
    InkParams ink;
    int min_area = 2;
    float bar_aspect = 3.f;      // w / h for a horizontal rule
    float bar_fill = 0.7f;
    float merge_overlap = 0.5f;  // x-overlap / narrower width for stacked parts
    int topk = 4;
    bool rescore = true;         // pass 2: re-rank candidates by prior + implied scale / baseline
    float scale_weight = 14.f;   // distance units per |log| scale mismatch
    float place_weight = 10.f;   // distance units per cap height of punctuation placement error
    float prior_weight = 10.f;   // with a unigram prior loaded: max penalty for the rarest symbol
};

// Unigram symbol prior (label codepoint -> count), e.g. data/math_prior.tsv written by
// `bench_latex --make-prior` from a *training / validation* split. Penalty = w * (1 - log(c+1)/log(max+1)).
struct MathPrior {
    std::unordered_map<char32_t, float> penalty;
    float unseen = 1.f;
    bool empty() const noexcept { return penalty.empty(); }

    static MathPrior from_counts(const std::unordered_map<char32_t, double>& counts) {
        MathPrior p;
        double mx = 1;
        for (const auto& [c, n] : counts) mx = std::max(mx, n);
        const double lm = std::log(mx + 1);
        for (const auto& [c, n] : counts) p.penalty[c] = float(1.0 - std::log(n + 1) / lm);
        return p;
    }
    bool load(const std::string& path) {
        std::ifstream in(path);
        if (!in) return false;
        std::unordered_map<char32_t, double> counts;
        unsigned long cp;
        double n;
        while (in >> cp >> n) counts[char32_t(cp)] = n;
        *this = from_counts(counts);
        return !empty();
    }
    float operator()(char32_t c) const {
        const auto it = penalty.find(c);
        return it == penalty.end() ? unseen : it->second;
    }
};

class MathReader {
public:
    MathReader(const ocr::Classifier& cls, MathReadParams p = {}, MathPrior prior = {})
        : cls_(cls), p_(p), prior_(std::move(prior)) {}

    std::vector<MathSymbol> symbols(GrayView crop) const {
        std::vector<MathSymbol> out;
        if (crop.width < 3 || crop.height < 3) return out;
        const ocr::CropContext C = ocr::analyze_crop(crop, p_.ink);
        const auto& comps = C.ccl.components;
        std::vector<uint32_t> keep;
        for (uint32_t i = 0; i < comps.size(); ++i)
            if (comps[i].area >= p_.min_area) keep.push_back(i);
        if (keep.empty()) return out;

        std::vector<int32_t> hs;
        for (uint32_t i : keep) hs.push_back(comps[i].bbox.h);
        std::nth_element(hs.begin(), hs.begin() + ptrdiff_t(hs.size() / 2), hs.end());
        const float Hmed = float(std::max(1, hs[hs.size() / 2]));

        auto is_bar = [&](const Component& c) {
            return float(c.bbox.w) >= p_.bar_aspect * float(c.bbox.h) && c.bbox.w >= 4 && c.fill() >= p_.bar_fill;
        };
        std::vector<int32_t> radical_start(comps.size(), -1);
        for (uint32_t i : keep) {
            const int32_t vx = radical_vinculum(C, comps[i]);
            if (vx < 0) continue;
            // A real radical covers a radicand: another component centred under the vinculum.
            const Rect& rb = comps[i].bbox;
            for (uint32_t j : keep)
                if (j != i && comps[j].bbox.x + comps[j].bbox.w / 2 >= vx &&
                    rb.contains_point(comps[j].bbox.x + comps[j].bbox.w / 2, comps[j].bbox.y + comps[j].bbox.h / 2)) {
                    radical_start[i] = vx;
                    break;
                }
        }

        // Stacked-part merging (union-find over kept components).
        std::vector<uint32_t> par(comps.size());
        std::iota(par.begin(), par.end(), 0u);
        auto find = [&](uint32_t i) { while (par[i] != i) i = par[i] = par[par[i]]; return i; };
        for (size_t a = 0; a < keep.size(); ++a)
            for (size_t b = a + 1; b < keep.size(); ++b) {
                const Component &A = comps[keep[a]], &B = comps[keep[b]];
                if (radical_start[keep[a]] >= 0 || radical_start[keep[b]] >= 0) continue;
                const Rect &ra = A.bbox, &rb = B.bbox;
                const int32_t xo = ra.x_overlap(rb), yo = ra.y_overlap(rb);
                const int32_t wmin = std::min(ra.w, rb.w);
                if (float(xo) < p_.merge_overlap * float(wmin)) continue;
                const int32_t gap = -yo;  // vertical gap (<= 0 means overlap)
                const bool bar_a = is_bar(A), bar_b = is_bar(B);
                bool merge = false;
                if (bar_a && bar_b) {  // = ≡ : similar-width rules close together
                    merge = gap >= 0 && float(gap) <= 0.6f * Hmed && float(std::max(ra.w, rb.w)) <= 1.25f * float(wmin);
                } else if (bar_a != bar_b) {  // ≤ ≥ ÷-bar: rule no wider than the glyph (a fraction bar is wider)
                    const Rect& bar = bar_a ? ra : rb;
                    const Rect& g = bar_a ? rb : ra;
                    merge = gap >= 0 && float(gap) <= 0.35f * Hmed && float(bar.w) <= 1.15f * float(g.w) && float(g.h) <= 1.2f * Hmed;
                } else {  // dots of i j ! ; : ? — one part tiny, or both tiny
                    const int32_t amin = std::min(A.area, B.area), amax = std::max(A.area, B.area);
                    const bool tiny_pair = float(std::max(ra.h, rb.h)) <= 0.4f * Hmed;
                    merge = gap >= -1 && float(gap) <= 0.45f * Hmed && (float(amin) <= 0.3f * float(amax) || tiny_pair);
                }
                if (merge) {
                    const uint32_t x = find(keep[a]), y = find(keep[b]);
                    if (x != y) par[std::max(x, y)] = std::min(x, y);
                }
            }
        std::vector<std::vector<uint32_t>> groups;
        {
            std::vector<int32_t> slot(comps.size(), -1);
            for (uint32_t i : keep) {
                const uint32_t r = find(i);
                if (slot[r] < 0) slot[r] = int32_t(groups.size()), groups.emplace_back();
                groups[size_t(slot[r])].push_back(i);
            }
        }

        for (const auto& g : groups) {
            MathSymbol s;
            std::vector<ocr::Source> src;
            for (uint32_t c : g) src.push_back({c, INT32_MIN, INT32_MAX}), s.box = s.box.unite(comps[c].bbox);
            if (g.size() == 1 && radical_start[g[0]] >= 0) {
                s.kind = MathSymbol::Kind::Radical;
                s.ch = 0x221A;
                s.tex = "\\sqrt";
                s.radicand_x = radical_start[g[0]];
                out.push_back(std::move(s));
                continue;
            }
            if (g.size() == 1 && is_bar(comps[g[0]])) {
                s.kind = MathSymbol::Kind::Bar;
                s.ch = U'-';
                s.tex = "-";
                // Minus width is ~1.14 cap heights in Computer Modern; its centre sits on the math axis
                // (~0.37 cap heights above the baseline).
                s.scale = float(s.box.w) / 1.14f;
                s.baseline = float(s.box.y) + 0.5f * float(s.box.h) + 0.37f * s.scale;
                out.push_back(std::move(s));
                continue;
            }
            const ocr::GlyphFeature f = ocr::sources_feature(C, src.data(), src.data() + src.size(), &s.box);
            s.ncand = int(cls_.classify(f, s.cand, size_t(std::min(p_.topk, 4))));
            if (s.ncand == 0) continue;
            s.ch = s.cand[0].ch;
            s.tex = tex_of(s.ch);
            s.dist = s.cand[0].dist;
            const ocr::Template& t = cls_.atlas().templates[s.cand[0].tmpl];
            const float rel = std::max(0.15f, t.bot_rel - t.top_rel);
            s.scale = float(s.box.h) / rel;
            s.baseline = float(s.box.bottom()) - t.bot_rel * s.scale;
            out.push_back(std::move(s));
        }
        if (p_.rescore) rescore(out);
        std::sort(out.begin(), out.end(), [](const MathSymbol& a, const MathSymbol& b) {
            return a.box.x != b.box.x ? a.box.x < b.box.x : a.box.y < b.box.y;
        });
        return out;
    }

    // Symbol frequency prior (distance units): letters, digits, common operators free; rare symbols pay.
    static float prior(char32_t c) {
        if ((c >= U'a' && c <= U'z') || (c >= U'A' && c <= U'Z') || (c >= U'0' && c <= U'9')) return 0.f;
        switch (c) {
            case U'(': case U')': case U'[': case U']': case U'+': case U'-': case U'=': case U',': case U'.':
            case U'|': case U'/': case U'\'': return 0.f;
            default: break;
        }
        if (c >= 0x03B1 && c <= 0x03C9) return 1.f;  // Greek
        if (c >= 0x0393 && c <= 0x03A9) return 1.f;
        switch (c) {
            case 0x2211: case 0x220F: case 0x222B: case 0x2202: case 0x221E: case 0x00B1: case 0x00D7: case 0x2264:
            case 0x2265: case 0x2192: case 0x2208: case 0x2261: case 0x2248: case 0x223C: case 0x22C5: case 0x2032:
            case U'<': case U'>': case U'{': case U'}': case U';': case U':': case U'!': case U'*': return 2.f;
            default: return 5.f;
        }
    }

private:
    static bool placement_punct(char32_t c) {
        return c == U',' || c == U'.' || c == U'\'' || c == 0x22C5 || c == 0x00B7 || c == 0x2032 || c == U'`';
    }

    // Pass 2. With the main scale S and baseline B of the formula (median of the larger symbols):
    //   * case twins / 1-l-I: the candidate whose implied scale fits S (on the baseline) or a script
    //     level 0.71 S / 0.5 S (off the baseline) wins;
    //   * punctuation (, . ' ·): placement against B decides;
    //   * a frequency prior keeps rare symbols from winning on near-ties.
    void rescore(std::vector<MathSymbol>& out) const {
        std::vector<float> sc;
        for (const auto& s : out)
            if (s.kind == MathSymbol::Kind::Glyph && s.scale > 0) sc.push_back(s.scale);
        if (sc.size() < 2) return;
        std::sort(sc.begin(), sc.end(), std::greater<float>());
        const float S = sc[(std::max<size_t>(1, (sc.size() * 6 + 9) / 10) - 1) / 2];  // median of the larger 60%
        std::vector<float> bl;
        for (const auto& s : out)
            if (s.kind == MathSymbol::Kind::Glyph && s.scale >= 0.84f * S) bl.push_back(s.baseline);
        if (bl.empty()) return;
        std::sort(bl.begin(), bl.end());
        const float B = bl[bl.size() / 2];
        for (auto& s : out) {
            if (s.kind != MathSymbol::Kind::Glyph || s.ncand == 0) continue;
            float best = 1e30f;
            int bi = 0;
            for (int c = 0; c < s.ncand; ++c) {
                const ocr::Template& t = cls_.atlas().templates[s.cand[c].tmpl];
                const float rel = std::max(0.15f, t.bot_rel - t.top_rel);
                const float sc_c = float(s.box.h) / rel;
                const float bl_c = float(s.box.bottom()) - t.bot_rel * sc_c;
                float pen;
                if (placement_punct(s.cand[c].ch)) {
                    // compare the symbol's top/bottom with where this punctuation sits on the main line
                    pen = (std::fabs(float(s.box.y) - (B + t.top_rel * S)) + std::fabs(float(s.box.bottom()) - (B + t.bot_rel * S))) / S;
                    pen *= p_.place_weight / p_.scale_weight;
                } else if (std::fabs(bl_c - B) <= 0.15f * S) {
                    pen = std::fabs(std::log(sc_c / S));
                } else {
                    pen = std::min(std::fabs(std::log(sc_c / (0.71f * S))), std::fabs(std::log(sc_c / (0.5f * S))));
                }
                const float pr = prior_.empty() ? prior(s.cand[c].ch) : p_.prior_weight * prior_(s.cand[c].ch);
                const float score = s.cand[c].dist + pr + p_.scale_weight * pen;
                if (score < best) best = score, bi = c;
            }
            if (bi != 0) {
                const ocr::Template& t = cls_.atlas().templates[s.cand[bi].tmpl];
                const float rel = std::max(0.15f, t.bot_rel - t.top_rel);
                s.ch = s.cand[bi].ch;
                s.tex = tex_of(s.ch);
                s.dist = s.cand[bi].dist;
                s.scale = float(s.box.h) / rel;
                s.baseline = float(s.box.bottom()) - t.bot_rel * s.scale;
            }
        }
    }

public:

private:
    // A radical is one component: a hook on the left whose ink reaches the bottom quarter, and a
    // horizontal vinculum in the top rows spanning most of the width. Returns the x where the
    // vinculum starts (the radicand's left edge), or -1.
    static int32_t radical_vinculum(const ocr::CropContext& C, const Component& c) {
        const Rect& b = c.bbox;
        if (b.h < 8 || b.w < b.h / 2 || c.fill() > 0.35f) return -1;
        const int32_t band = std::max(1, b.h / 8);
        int32_t best_len = 0, best_x0 = -1;
        bool hook_low = false;
        for (uint32_t i = c.run_begin; i < c.run_end; ++i) {
            const Run& r = C.ccl.runs[i];
            if (r.y < b.y + band && r.x1 - r.x0 > best_len) best_len = r.x1 - r.x0, best_x0 = r.x0;
            if (r.y >= b.bottom() - b.h / 4 && r.x0 < b.x + b.w * 2 / 5) hook_low = true;
        }
        if (!hook_low || float(best_len) < 0.55f * float(b.w) || best_x0 <= b.x) return -1;
        return best_x0;
    }

    const ocr::Classifier& cls_;
    MathReadParams p_;
    MathPrior prior_;
};

}  // namespace dks::latex
