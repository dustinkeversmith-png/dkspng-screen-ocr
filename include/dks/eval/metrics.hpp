#pragma once
// Evaluation metrics: box matching (IoU), text grouping split/merge, containment depth, CER / EM.
#include <algorithm>
#include <cstdint>
#include <numeric>
#include <string>
#include <vector>

#include "../core/geometry.hpp"
#include "../core/utf8.hpp"
#include "../segment/hierarchy.hpp"

namespace dks::eval {

// ------------------------------------------------------------------------------------ detection
struct MatchResult {
    std::vector<int32_t> gt_to_pred;  // -1 = unmatched
    std::vector<float> gt_best_iou;   // best IoU against any prediction (not necessarily matched)
    size_t matched = 0;
};

// Greedy one-to-one matching in descending IoU order (standard for detection eval).
inline MatchResult match_boxes(const std::vector<Rect>& gt, const std::vector<Rect>& pred, float thr) {
    MatchResult m;
    m.gt_to_pred.assign(gt.size(), -1);
    m.gt_best_iou.assign(gt.size(), 0.f);
    struct Cand { float iou; uint32_t g, p; };
    std::vector<Cand> cands;
    // Sweep on x to avoid the full N*M product on screens with thousands of boxes.
    std::vector<uint32_t> po(pred.size());
    std::iota(po.begin(), po.end(), 0u);
    std::sort(po.begin(), po.end(), [&](uint32_t a, uint32_t b) { return pred[a].x < pred[b].x; });
    for (uint32_t g = 0; g < gt.size(); ++g) {
        const Rect& G = gt[g];
        for (uint32_t p : po) {
            if (pred[p].x >= G.right()) break;
            if (pred[p].right() <= G.x) continue;
            const float iou = G.iou(pred[p]);
            if (iou <= 0.f) continue;
            m.gt_best_iou[g] = std::max(m.gt_best_iou[g], iou);
            if (iou >= thr) cands.push_back({iou, g, p});
        }
    }
    std::sort(cands.begin(), cands.end(), [](const Cand& a, const Cand& b) {
        return a.iou != b.iou ? a.iou > b.iou : (a.g != b.g ? a.g < b.g : a.p < b.p);
    });
    std::vector<uint8_t> used(pred.size(), 0);
    for (const Cand& c : cands) {
        if (m.gt_to_pred[c.g] >= 0 || used[c.p]) continue;
        m.gt_to_pred[c.g] = int32_t(c.p);
        used[c.p] = 1;
        ++m.matched;
    }
    return m;
}

struct DetectionScore {
    size_t n_gt = 0, n_pred = 0, tp50 = 0, tp75 = 0;
    double sum_best_iou = 0;
    void add(const std::vector<Rect>& gt, const std::vector<Rect>& pred) {
        const MatchResult m50 = match_boxes(gt, pred, 0.5f), m75 = match_boxes(gt, pred, 0.75f);
        n_gt += gt.size(), n_pred += pred.size(), tp50 += m50.matched, tp75 += m75.matched;
        for (float v : m50.gt_best_iou) sum_best_iou += v;
    }
    double recall50() const { return n_gt ? double(tp50) / n_gt : 0; }
    double recall75() const { return n_gt ? double(tp75) / n_gt : 0; }
    double precision50() const { return n_pred ? double(tp50) / n_pred : 0; }
    double precision75() const { return n_pred ? double(tp75) / n_pred : 0; }
    double f1_50() const { const double p = precision50(), r = recall50(); return p + r > 0 ? 2 * p * r / (p + r) : 0; }
    double mean_best_iou() const { return n_gt ? sum_best_iou / n_gt : 0; }
};

// ------------------------------------------------------------------------------------ grouping
// A prediction "belongs" to a GT box when >= `own` of the prediction's area lies inside it.
//   split : GT covered by >1 predictions         merge : prediction covering >1 GT boxes
//   miss  : GT with no belonging prediction      (GT counted as "covered" if >= own of *its* area is in P)
struct GroupingScore {
    size_t n_gt = 0, n_pred = 0, ok = 0, split = 0, miss = 0, merge = 0;
    double sum_overlap = 0;  // per GT: area covered by its belonging predictions / GT area
    void add(const std::vector<Rect>& gt, const std::vector<Rect>& pred, float own = 0.5f) {
        n_gt += gt.size(), n_pred += pred.size();
        for (const Rect& G : gt) {
            int belong = 0;
            int64_t covered = 0;
            for (const Rect& P : pred) {
                const int64_t i = G.inter_area(P);
                if (i > 0 && double(i) >= own * double(P.area())) ++belong, covered += i;
            }
            if (belong == 0) ++miss; else if (belong == 1) ++ok; else ++split;
            sum_overlap += G.area() ? std::min(1.0, double(covered) / double(G.area())) : 0;
        }
        for (const Rect& P : pred) {
            int covers = 0;
            for (const Rect& G : gt) {
                const int64_t i = G.inter_area(P);
                if (i > 0 && double(i) >= own * double(G.area())) ++covers;
            }
            if (covers > 1) ++merge;
        }
    }
    double split_rate() const { return n_gt ? double(split) / n_gt : 0; }
    double miss_rate() const { return n_gt ? double(miss) / n_gt : 0; }
    double one_to_one() const { return n_gt ? double(ok) / n_gt : 0; }
    double merge_rate() const { return n_pred ? double(merge) / n_pred : 0; }
    double overlap_ratio() const { return n_gt ? sum_overlap / n_gt : 0; }
};

// ------------------------------------------------------------------------------------ hierarchy
// Of the GT boxes matched at IoU>=0.5, how many sit at the same containment depth in both forests,
// and how many have their matched parent correspond (parent(pred) == match(parent(gt))).
struct DepthScore {
    size_t matched = 0, same_depth = 0, parent_ok = 0;
    void add(const std::vector<Rect>& gt, const std::vector<Rect>& pred) {
        const Forest fg = build_containment_forest(gt), fp = build_containment_forest(pred);
        const MatchResult m = match_boxes(gt, pred, 0.5f);
        for (size_t g = 0; g < gt.size(); ++g) {
            const int32_t p = m.gt_to_pred[g];
            if (p < 0) continue;
            ++matched;
            if (fg.depth[g] == fp.depth[size_t(p)]) ++same_depth;
            const int32_t gp = fg.parent[g], pp = fp.parent[size_t(p)];
            if ((gp < 0 && pp < 0) || (gp >= 0 && m.gt_to_pred[size_t(gp)] == pp && pp >= 0)) ++parent_ok;
        }
    }
    double depth_precision() const { return matched ? double(same_depth) / matched : 0; }
    double parent_precision() const { return matched ? double(parent_ok) / matched : 0; }
};

// ------------------------------------------------------------------------------------ text
inline size_t levenshtein(const std::u32string& a, const std::u32string& b) {
    std::vector<size_t> prev(b.size() + 1), cur(b.size() + 1);
    std::iota(prev.begin(), prev.end(), size_t(0));
    for (size_t i = 1; i <= a.size(); ++i) {
        cur[0] = i;
        for (size_t j = 1; j <= b.size(); ++j)
            cur[j] = std::min({prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] == b[j - 1] ? 0 : 1)});
        std::swap(prev, cur);
    }
    return prev[b.size()];
}

inline std::u32string fold_case(std::u32string s) {
    for (char32_t& c : s) {
        if (c >= U'A' && c <= U'Z') c = c - U'A' + U'a';
        else if (c >= 0xC0 && c <= 0xDE && c != 0xD7) c += 0x20;  // Latin-1 upper -> lower
    }
    return s;
}

struct TextScore {
    size_t n = 0, exact = 0, exact_ci = 0, edits = 0, edits_ci = 0, chars = 0;
    void add(const std::string& gt_utf8, const std::string& pred_utf8) {
        const std::u32string g = utf8_decode(gt_utf8), p = utf8_decode(pred_utf8);
        const std::u32string gf = fold_case(g), pf = fold_case(p);
        ++n;
        chars += g.size();
        const size_t e = levenshtein(g, p), ec = levenshtein(gf, pf);
        edits += e, edits_ci += ec;
        exact += e == 0, exact_ci += ec == 0;
    }
    double cer() const { return chars ? double(edits) / chars : 0; }
    double cer_ci() const { return chars ? double(edits_ci) / chars : 0; }
    double em() const { return n ? double(exact) / n : 0; }
    double em_ci() const { return n ? double(exact_ci) / n : 0; }
};

}  // namespace dks::eval
