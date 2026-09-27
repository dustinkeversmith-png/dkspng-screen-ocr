#pragma once
// Word / line recogniser.
//
//   crop -> ink (Otsu levels + border/run-length polarity, 50% coverage mask) -> CCL
//        -> atoms (vertical stacks merged: i j : ; = ä; small enclosed parts)
//        -> trim fragments of neighbouring text clipped by a loose crop
//        -> pieces (wide atoms cut at column-profile minima: touching glyphs)
//        -> hypotheses: every run of 1..max_merge consecutive pieces, classified once (top-k chars)
//        -> DP pass 1: cost = shape distance * (width / H) + prior(char) + height consistency + penalty
//        -> fit baseline B and cap height S from placement-informative glyphs (robust LSQ)
//        -> DP pass 2: same, with placement error vs (B, S) -> segmentation + labels jointly
//           (fixes o/O, s/S, ,/', '.'/'•', and stops merged glyph runs from matching '~', '"', 'm')
//        -> spaces from inter-glyph gaps, then per-word case / digit consistency (no dictionary)
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include "../core/utf8.hpp"
#include "classifier.hpp"

namespace dks::ocr {

struct RecognizerParams {
    float char_penalty = 10.0f;     // cost per emitted glyph (distance units), discourages over-splitting
    float skip_floor = 18.0f;       // cost to drop a speck (must exceed char_penalty + a clean '.')
    float skip_cost = 400.0f;       // cost to drop a glyph-sized piece (effectively never)
    float speck_h = 0.3f;           // pieces shorter than this * H count as specks
    int max_merge = 5;              // pieces per glyph hypothesis
    float max_glyph_aspect = 1.9f;  // hypothesis width / text height
    float cut_min_aspect = 0.62f;   // atoms wider than this * H are considered for cutting
    float cut_profile = 0.55f;      // cut column ink <= this * atom max column ink (DP arbitrates)
    float merge_gap = 0.22f;        // pieces separated by more than this * H never form one glyph
    bool spaces = true;
    float space_gap = 0.38f;        // gap / cap height that becomes a space
    bool line_metrics = true;
    bool harmonize = true;          // lexical case / digit consistency per word
    bool trim_border = true;        // drop fragments of neighbouring text cut by a loose crop
    bool trace = false;             // dump every hypothesis to stderr
    float metric_weight = 20.f;     // distance units per cap-height of placement error (pass 2)
    float height_weight = 20.f;     // pass 1: |glyph height / H - template height in cap units|
    bool width_norm = true;         // DP cost = distance * (glyph width / H) + char_penalty
    bool ascii_punct = false;       // fold typographic punctuation to ASCII (» -> >>, ’ -> ', – -> -)
    InkParams ink;                  // query-side binarisation options (templates always use defaults)
    int topk = 6;
    int min_contrast = 28;
};

struct GlyphResult {
    Rect box;
    GlyphFeature feat;
    float wn = 1.f;  // shape-cost weight: glyph width / text height (see RecognizerParams::width_norm)
    Candidate cand[8];
    int ncand = 0;
    int chosen = 0;
    char32_t ch() const noexcept { return ncand ? cand[chosen].ch : U'?'; }
};

struct Recognition {
    std::u32string text;
    std::vector<GlyphResult> glyphs;
    float cost = 0;
    float baseline = 0, cap_height = 0;
    bool line_fit = false;      // a baseline/cap-height model could be fitted
    // Evidence that the crop really is text (see ocr::text_confidence):
    float mean_dist = 0;        // mean shape distance of the chosen glyph labels
    float alnum_frac = 0;       // share of glyphs that are letters or digits
    float fit_residual = 0;     // mean placement error vs the fitted line, in cap heights
    std::string utf8() const { return utf8_encode(text); }
};

class Recognizer {
public:
    Recognizer(const Classifier& cls, RecognizerParams p = {}) : cls_(cls), p_(p) {}
    RecognizerParams& params() noexcept { return p_; }

    Recognition recognize(GrayView crop) const {
        Recognition R;
        if (crop.width < 2 || crop.height < 4) return R;
        const CropContext C = analyze_crop(crop, p_.ink);
        if (C.ink.contrast() < p_.min_contrast) return R;
        const auto& comps = C.ccl.components;
        if (comps.empty()) return R;

        // Text height H: robust max (median of the tallest third) of non-speck components.
        std::vector<int32_t> hs;
        for (const auto& c : comps) if (c.area >= 3) hs.push_back(c.bbox.h);
        if (hs.empty()) return R;
        std::sort(hs.begin(), hs.end());
        const float H = float(hs[hs.size() - 1 - (hs.size() / 3) / 2]);

        // --- atoms
        std::vector<uint32_t> ord(comps.size());
        std::iota(ord.begin(), ord.end(), 0u);
        std::vector<uint32_t> par(comps.size());
        std::iota(par.begin(), par.end(), 0u);
        auto find = [&](uint32_t i) { while (par[i] != i) i = par[i] = par[par[i]]; return i; };
        for (uint32_t a = 0; a < comps.size(); ++a) {
            for (uint32_t b = a + 1; b < comps.size(); ++b) {
                const Rect &A = comps[a].bbox, &B = comps[b].bbox;
                const int32_t xo = A.x_overlap(B), yo = A.y_overlap(B);
                const bool stacked = xo >= std::max(1, std::min(A.w, B.w) / 2) && yo <= 0 && -yo <= int32_t(0.45f * H);
                const bool enclosed = (A.contains(B) && float(A.h) <= 1.5f * H) || (B.contains(A) && float(B.h) <= 1.5f * H);
                if (stacked || enclosed) {
                    const uint32_t ra = find(a), rb = find(b);
                    if (ra != rb) par[std::max(ra, rb)] = std::min(ra, rb);
                }
            }
        }
        std::vector<std::vector<uint32_t>> atoms;
        {
            // Single-pixel components are specks unless they stacked onto something (a 1px i/j dot
            // is common in small ClearType / hinted UI text).
            std::vector<int32_t> slot(comps.size(), -1), members(comps.size(), 0);
            for (uint32_t i = 0; i < comps.size(); ++i) ++members[find(i)];
            for (uint32_t i = 0; i < comps.size(); ++i) {
                if (comps[i].area < 2 && members[find(i)] < 2) continue;
                const uint32_t r = find(i);
                if (slot[r] < 0) slot[r] = int32_t(atoms.size()), atoms.emplace_back();
                atoms[size_t(slot[r])].push_back(i);
            }
        }

        if (p_.trim_border) trim_border_fragments(atoms, comps, C.width, C.height, H);
        if (atoms.empty()) return R;

        // --- pieces
        struct Piece { std::vector<Source> src; Rect box; };
        std::vector<Piece> pieces;
        for (const auto& atom : atoms) {
            Rect ab;
            for (uint32_t c : atom) ab = ab.unite(comps[c].bbox);
            std::vector<int32_t> cuts;
            if (float(ab.w) > p_.cut_min_aspect * H && ab.w >= 4) {
                std::vector<int32_t> prof(size_t(ab.w), 0);
                for (uint32_t c : atom)
                    for (uint32_t i = comps[c].run_begin; i < comps[c].run_end; ++i) {
                        const Run& r = C.ccl.runs[i];
                        for (int32_t x = r.x0; x < r.x1; ++x) ++prof[size_t(x - ab.x)];
                    }
                const int32_t mx = *std::max_element(prof.begin(), prof.end());
                const int32_t edge = std::max<int32_t>(1, int32_t(0.12f * H));
                for (int32_t x = edge; x < ab.w - edge; ++x) {
                    const int32_t v = prof[size_t(x)];
                    if (float(v) > p_.cut_profile * float(mx)) continue;
                    if (v > prof[size_t(x - 1)] || v > prof[size_t(x + 1)]) continue;
                    if (!cuts.empty() && float(x - cuts.back()) < std::max(2.f, 0.15f * H)) {  // keep the lowest nearby
                        if (v <= prof[size_t(cuts.back())]) cuts.back() = x;
                        continue;
                    }
                    cuts.push_back(x);
                }
            }
            int32_t prev = INT32_MIN;
            cuts.push_back(INT32_MAX);
            for (int32_t cx : cuts) {
                const int32_t xlo = prev == INT32_MIN ? INT32_MIN : ab.x + prev;
                const int32_t xhi = cx == INT32_MAX ? INT32_MAX : ab.x + cx;
                Piece pc;
                for (uint32_t c : atom) pc.src.push_back({c, xlo, xhi});
                pc.box = sources_bbox(C, pc.src.data(), pc.src.data() + pc.src.size());
                if (!pc.box.empty()) pieces.push_back(std::move(pc));
                prev = cx;
            }
        }
        std::sort(pieces.begin(), pieces.end(), [](const Piece& a, const Piece& b) {
            return a.box.cx2() != b.box.cx2() ? a.box.cx2() < b.box.cx2() : a.box.y < b.box.y;
        });
        const size_t n = pieces.size();
        if (n == 0) return R;

        // --- glyph hypotheses: every run of 1..max_merge consecutive pieces, classified once and
        //     reused by both DP passes.
        const size_t M = size_t(p_.max_merge);
        std::vector<GlyphResult> hyp(n * M);
        std::vector<uint8_t> ok(n * M, 0);
        std::vector<Source> src;
        for (size_t i = 0; i < n; ++i) {
            src.clear();
            Rect ub;
            for (size_t j = i; j < n && j < i + M; ++j) {
                if (j > i && float(pieces[j].box.x - ub.right()) > p_.merge_gap * H) break;
                ub = ub.unite(pieces[j].box);
                src.insert(src.end(), pieces[j].src.begin(), pieces[j].src.end());
                if (float(ub.w) > p_.max_glyph_aspect * H) break;
                GlyphResult& g = hyp[i * M + (j - i)];
                g.feat = sources_feature(C, src.data(), src.data() + src.size(), &g.box);
                g.ncand = int(cls_.classify(g.feat, g.cand, size_t(std::min(p_.topk, 8))));
                ok[i * M + (j - i)] = g.ncand > 0;
                if (p_.trace && g.ncand)
                    std::fprintf(stderr, "  hyp pieces[%zu..%zu] box %d,%d %dx%d  best U+%04X %.1f\n", i, j, g.box.x, g.box.y,
                                 g.box.w, g.box.h, unsigned(g.cand[0].ch), g.cand[0].dist);
            }
        }

        std::vector<float> skip(n);
        for (size_t i = 0; i < n; ++i) {
            const float r = std::min(1.f, float(pieces[i].box.h) / H);
            // Specks (< speck_h * H) are cheap to drop; anything glyph-sized must be read (a wrong
            // label costs the same one edit as a deletion, and is often right).
            skip[i] = r < p_.speck_h ? p_.skip_floor : p_.skip_cost;
        }

        // Pass 1: shape (+ prior) only.
        const float NaN = std::numeric_limits<float>::quiet_NaN();
        auto path = run_dp(n, hyp, ok, skip, H, [&](GlyphResult& g) { return choose(g, NaN, NaN, H); }, R.cost);
        for (size_t h : path) R.glyphs.push_back(hyp[h]);
        if (R.glyphs.empty()) return R;
        R.cap_height = H;

        // Pass 2: fit the line (baseline B, cap height S) on placement-informative glyphs, then redo
        // the DP with placement-aware costs. Segmentation and labels are decided jointly.
        float B = 0, S = 0;
        if (p_.line_metrics && fit_robust(R.glyphs, B, S)) {
            path = run_dp(n, hyp, ok, skip, H, [&](GlyphResult& g) { return choose(g, B, S); }, R.cost);
            R.glyphs.clear();
            for (size_t h : path) R.glyphs.push_back(hyp[h]);
            float B2, S2;
            if (fit_robust(R.glyphs, B2, S2)) {
                B = B2, S = S2;
                for (auto& g : R.glyphs) choose(g, B, S);
            }
            R.baseline = B;
            R.cap_height = S;
            R.line_fit = true;
        }

        // --- text: spaces from gaps, then lexical case / digit consistency per word
        const float Sx = R.cap_height > 0 ? R.cap_height : H;
        std::vector<char32_t> word;
        auto flush = [&] {
            if (p_.harmonize) harmonize(word);
            R.text.append(word.begin(), word.end());
            word.clear();
        };
        for (size_t k = 0; k < R.glyphs.size(); ++k) {
            if (k > 0 && p_.spaces) {
                const int32_t gap = R.glyphs[k].box.x - R.glyphs[k - 1].box.right();
                if (float(gap) > p_.space_gap * Sx) { flush(); R.text.push_back(U' '); }
            }
            char32_t c = R.glyphs[k].ch();
            if (c == 0x131) c = U'i';  // dotless i: an i whose faint dot fell below 50% coverage
            if (p_.ascii_punct) {
                if (c == 0xBB || c == 0xAB) { word.push_back(c == 0xBB ? U'>' : U'<'); c = word.back(); }
                else if (c == 0x2019 || c == 0x2018) c = U'\'';
                else if (c == 0x201C || c == 0x201D) c = U'"';
                else if (c == 0x2013 || c == 0x2014) c = U'-';
            }
            word.push_back(c);
        }
        flush();

        // Text evidence.
        float sd = 0, pe = 0;
        int an = 0;
        for (const auto& g : R.glyphs) {
            sd += g.cand[g.chosen].dist;
            const char32_t c = g.ch();
            an += (c >= U'a' && c <= U'z') || (c >= U'A' && c <= U'Z') || (c >= U'0' && c <= U'9') || (c >= 0xC0 && c <= 0x17F);
            if (R.line_fit) pe += placement_error(g, g.chosen, R.baseline, R.cap_height);
        }
        const float ng = float(std::max<size_t>(1, R.glyphs.size()));
        R.mean_dist = sd / ng;
        R.alnum_frac = float(an) / ng;
        R.fit_residual = R.line_fit ? pe / ng : 1.f;
        return R;
    }

private:
    // A loose crop (detector box, quantised GT) clips neighbours: the top/bottom of adjacent lines
    // and slivers of adjacent words. Drop atoms that touch the crop frame *and* sit outside the main
    // text band (top/bottom) or are thin cut slivers (left/right).
    static void trim_border_fragments(std::vector<std::vector<uint32_t>>& atoms, const std::vector<Component>& comps,
                                      int32_t W, int32_t Hc, float H) {
        std::vector<Rect> box(atoms.size());
        std::vector<int32_t> tops, bots;
        for (size_t a = 0; a < atoms.size(); ++a) {
            for (uint32_t c : atoms[a]) box[a] = box[a].unite(comps[c].bbox);
            if (float(box[a].h) >= 0.5f * H && box[a].y > 0 && box[a].bottom() < Hc)
                tops.push_back(box[a].y), bots.push_back(box[a].bottom());
        }
        if (tops.empty()) return;
        std::sort(tops.begin(), tops.end());
        std::sort(bots.begin(), bots.end());
        const float band_t = float(tops[tops.size() / 2]), band_b = float(bots[bots.size() / 2]);
        std::vector<std::vector<uint32_t>> kept;
        for (size_t a = 0; a < atoms.size(); ++a) {
            const Rect& b = box[a];
            const float cy = 0.5f * float(b.y + b.bottom());
            const bool touches_tb = b.y == 0 || b.bottom() == Hc;
            const bool touches_lr = b.x == 0 || b.right() == W;
            if (touches_tb && (cy < band_t || cy > band_b)) continue;
            if (touches_lr && float(b.w) < 0.3f * H && float(b.h) >= 0.5f * H) {
                // Only a sliver separated from the rest by a word-sized gap is a neighbour fragment.
                int32_t gap = INT32_MAX;
                for (size_t o = 0; o < atoms.size(); ++o) {
                    if (o == a || box[o].empty()) continue;
                    const int32_t g = b.x == 0 ? box[o].x - b.right() : b.x - box[o].right();
                    if (g >= -1) gap = std::min(gap, g);
                }
                if (float(gap) > 0.3f * H) continue;
            }
            kept.push_back(atoms[a]);
        }
        atoms.swap(kept);
    }

    // Glyphs whose vertical placement is not informative about the line (case twins, punctuation).
    static bool placement_ambiguous(char32_t c) noexcept {
        static const std::u32string amb = U"cCoOsSvVwWxXzZuU0.,:;'\"`-_~=+*^°·•’“”|";
        return amb.find(c) != std::u32string::npos;
    }

    // Least-squares fit of (B, S) from   top = B + top_rel*S,  bottom = B + bot_rel*S.
    static bool fit_line(const std::vector<GlyphResult>& gs, const std::vector<uint8_t>& use, const Atlas& A,
                         float& B, float& S) {
        double n = 0, sr = 0, srr = 0, sy = 0, sry = 0;
        for (size_t k = 0; k < gs.size(); ++k) {
            if (!use[k]) continue;
            const Template& t = A.templates[gs[k].cand[gs[k].chosen].tmpl];
            const double obs[2] = {double(gs[k].box.y), double(gs[k].box.bottom())};
            const double rel[2] = {double(t.top_rel), double(t.bot_rel)};
            for (int e = 0; e < 2; ++e) n += 1, sr += rel[e], srr += rel[e] * rel[e], sy += obs[e], sry += rel[e] * obs[e];
        }
        const double det = n * srr - sr * sr;
        if (n < 2 || std::fabs(det) < 1e-6) return false;
        S = float((n * sry - sr * sy) / det);
        B = float((sy - double(S) * sr) / n);
        return S > 2.f;
    }

    float placement_error(const GlyphResult& g, int c, float B, float S) const {
        const Template& t = cls_.atlas().templates[g.cand[c].tmpl];
        return (std::fabs(float(g.box.y) - (B + t.top_rel * S)) + std::fabs(float(g.box.bottom()) - (B + t.bot_rel * S))) / S;
    }

    // Frequency prior (distance units): letters/digits free, common punctuation cheap, rare symbols
    // expensive. Stops merged junk from matching '~', '^', '=' etc.
    static float prior(char32_t c) noexcept {
        if ((c >= U'a' && c <= U'z') || (c >= U'A' && c <= U'Z') || (c >= U'0' && c <= U'9') || c == 0x131) return 0.f;
        static const std::u32string common = U".,:;-/'()@&%$!?\"#+*€£’";
        if (common.find(c) != std::u32string::npos) return 1.5f;
        if (c >= 0xC0 && c <= 0xFF && c != 0xD7) return 2.5f;  // accented Latin-1 letters
        return 6.f;
    }

    // Best candidate for a hypothesis; with a line fit (B,S) the placement error is added.
    // Without a fit (pass 1) only the relative height is checked against H (tallest-text estimate).
    float choose(GlyphResult& g, float B, float S, float H = 0.f) const {
        float best = std::numeric_limits<float>::infinity();
        for (int c = 0; c < g.ncand; ++c) {
            float s = g.cand[c].dist * g.wn + prior(g.cand[c].ch);
            if (S == S) s += p_.metric_weight * placement_error(g, c, B, S);  // S == S: not NaN
            else if (H > 0.f) {
                const Template& t = cls_.atlas().templates[g.cand[c].tmpl];
                s += p_.height_weight * std::max(0.f, std::fabs(float(g.box.h) / H - (t.bot_rel - t.top_rel)) - 0.25f);
            }
            if (s < best) best = s, g.chosen = c;
        }
        return best;
    }

    template <class Cost>
    std::vector<size_t> run_dp(size_t n, std::vector<GlyphResult>& hyp, const std::vector<uint8_t>& ok,
                               const std::vector<float>& skip, float Hn, Cost cost, float& total) const {
        const size_t M = size_t(p_.max_merge);
        const float INF = std::numeric_limits<float>::infinity();
        std::vector<float> best(n + 1, INF);
        std::vector<int64_t> from(n + 1, -1);  // >=0: hyp index; <=-2: skipped piece -(i)-2
        best[0] = 0;
        for (size_t i = 0; i < n; ++i) {
            if (best[i] == INF) continue;
            if (best[i] + skip[i] < best[i + 1]) best[i + 1] = best[i] + skip[i], from[i + 1] = -int64_t(i) - 2;
            for (size_t k = 0; k < M && i + k < n; ++k) {
                const size_t h = i * M + k;
                if (!ok[h]) continue;
                // The shape distance is weighted by the hypothesis width in text heights so that
                // covering more ink with one glyph is not cheaper than explaining it with several.
                // Prior / placement terms are per glyph and are not scaled.
                hyp[h].wn = p_.width_norm ? std::max(0.35f, float(hyp[h].box.w) / Hn) : 1.f;
                const float c = best[i] + cost(hyp[h]) + p_.char_penalty;
                if (c < best[i + k + 1]) best[i + k + 1] = c, from[i + k + 1] = int64_t(h);
            }
        }
        total = best[n];
        std::vector<size_t> path;
        for (size_t j = n; j > 0;) {
            const int64_t f = from[j];
            if (f <= -2) { j = size_t(-f - 2); continue; }
            if (f < 0) break;
            path.push_back(size_t(f));
            j = size_t(f) / M;
        }
        std::reverse(path.begin(), path.end());
        for (size_t h : path) cost(hyp[h]);  // re-apply each path hypothesis' candidate choice
        return path;
    }

    bool fit_robust(const std::vector<GlyphResult>& gs, float& B, float& S) const {
        std::vector<uint8_t> use(gs.size(), 0);
        for (size_t k = 0; k < gs.size(); ++k) use[k] = !placement_ambiguous(gs[k].ch());
        if (!fit_line(gs, use, cls_.atlas(), B, S)) return false;
        for (size_t k = 0; k < gs.size(); ++k)
            use[k] = use[k] && placement_error(gs[k], gs[k].chosen, B, S) < 0.35f;
        float B2, S2;
        if (fit_line(gs, use, cls_.atlas(), B2, S2)) B = B2, S = S2;
        return true;
    }

    // Deterministic lexical consistency inside one word (no dictionary):
    //   * digit context: O/o -> 0, l/I/| -> 1   * letter context: 0 -> O/o, 1 -> l/I
    //   * case twins (c o s v w x z u, I/l) follow the case of the word's unambiguous letters.
    static void harmonize(std::vector<char32_t>& w) {
        auto is_digit = [](char32_t c) { return c >= U'0' && c <= U'9'; };
        auto is_up = [](char32_t c) { return c >= U'A' && c <= U'Z'; };
        auto is_lo = [](char32_t c) { return c >= U'a' && c <= U'z'; };
        static const std::u32string twins = U"cosvwxzuCOSVWXZU";
        auto twin = [&](char32_t c) { return twins.find(c) != std::u32string::npos; };
        auto oneish = [](char32_t c) { return c == U'l' || c == U'I' || c == U'|' || c == U'1'; };
        auto zeroish = [](char32_t c) { return c == U'o' || c == U'O' || c == U'0'; };

        int up = 0, lo = 0, dig = 0;
        for (char32_t c : w) {
            if (is_digit(c) && c != U'0' && c != U'1') ++dig;
            else if (is_up(c) && !twin(c) && c != U'I') ++up;
            else if (is_lo(c) && !twin(c) && c != U'l') ++lo;
        }
        for (size_t i = 0; i < w.size(); ++i) {
            char32_t& c = w[i];
            if (dig > 0 && up + lo == 0) {  // numeric token
                if (zeroish(c)) c = U'0';
                else if (oneish(c)) c = U'1';
                else if (c == U'S' || c == U's') c = U'5';
                continue;
            }
            if (c == U'5' && dig == 1 && up + lo >= 2) c = (lo >= up) ? U's' : U'S';  // lone 5 among letters
            if (up + lo == 0) continue;
            const bool lower = lo >= up;
            if (c == U'0') c = lower ? U'o' : U'O';
            else if (c == U'1' && (up + lo) > dig) c = lower ? U'l' : U'I';
            else if (twin(c)) {
                if (lower && is_up(c) && !(i == 0 && lo > 0)) c = c - U'A' + U'a';  // keep a capitalised initial
                else if (!lower && is_lo(c)) c = c - U'a' + U'A';
            } else if ((c == 0xCD || c == 0xCC || c == 0xCE || c == 0xCF) && lower && i > 0) {
                c = U'i';  // a 1px i-dot and an acute over a 1px stem are the same raster
            } else if (c == U'I' && lower && i > 0) c = U'l';
            else if (c == U'l' && !lower) c = U'I';
        }
    }

    const Classifier& cls_;
    RecognizerParams p_;
};

}  // namespace dks::ocr
