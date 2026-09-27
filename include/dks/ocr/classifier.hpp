#pragma once
// Exact nearest-template search returning the best k *distinct characters*.
//
//   SearchMode::Sorted  (default) filter & refine: a vectorised lower bound (4x4 block sums of the
//                       bitmap + the scalar terms) over all templates, then best-band-first exact
//                       refinement of every template whose bound is under the admission threshold.
//   SearchMode::VPTree  vantage-point tree over the full metric. Kept for comparison: in this 256-D
//                       space it degenerates to ~15-25k distance evaluations per query (vs ~1.3k).
//                       (A flat ball partition was also tried; it needed 7k-38k balls and was slower.)
// Both modes are exact and deterministic (ties broken by template index).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <unordered_map>
#include <numeric>
#include <vector>

#include "atlas.hpp"

namespace dks::ocr {

struct Candidate {
    char32_t ch = 0;
    float dist = std::numeric_limits<float>::infinity();
    uint32_t tmpl = UINT32_MAX;
};

enum class SearchMode : uint8_t { Sorted, VPTree };

// Keeps the k best distinct characters; tau() is the admission bound for pruning.
class TopChars {
public:
    // Keep at most k characters, and only those within `margin` of the best one.
    TopChars(size_t k, float margin) : k_(k), margin_(margin) {}
    float tau() const noexcept {
        if (n_ == 0) return std::numeric_limits<float>::infinity();
        const float m = c_[0].dist + margin_;
        return n_ < k_ ? m : std::min(m, c_[n_ - 1].dist);
    }
    void offer(char32_t ch, float d, uint32_t t) noexcept {
        for (size_t i = 0; i < n_; ++i) {
            if (c_[i].ch != ch) continue;
            if (d < c_[i].dist || (d == c_[i].dist && t < c_[i].tmpl)) { c_[i].dist = d, c_[i].tmpl = t; bubble(i); }
            return;
        }
        if (n_ < k_) c_[n_++] = {ch, d, t};
        else if (d < c_[n_ - 1].dist) c_[n_ - 1] = {ch, d, t};
        else return;
        bubble(n_ - 1);
    }
    size_t size() const noexcept {  // entries that are still within the margin of the best
        size_t k = 0;
        while (k < n_ && c_[k].dist <= c_[0].dist + margin_) ++k;
        return k;
    }
    const Candidate& operator[](size_t i) const noexcept { return c_[i]; }

private:
    void bubble(size_t i) noexcept {
        while (i > 0 && (c_[i].dist < c_[i - 1].dist || (c_[i].dist == c_[i - 1].dist && c_[i].tmpl < c_[i - 1].tmpl)))
            std::swap(c_[i], c_[i - 1]), --i;
    }
    static constexpr size_t kMax = 16;
    Candidate c_[kMax];
    size_t k_, n_ = 0;
    float margin_;
};

class Classifier {
public:
    Classifier(const Atlas& atlas, FeatureWeights w = {}, SearchMode mode = SearchMode::Sorted, float margin = 30.f)
        : atlas_(atlas), w_(w), mode_(mode), margin_(margin) {
        const size_t n = atlas.templates.size();
        idx_->by_aspect.resize(n);
        std::iota(idx_->by_aspect.begin(), idx_->by_aspect.end(), 0u);
        std::sort(idx_->by_aspect.begin(), idx_->by_aspect.end(), [&](uint32_t a, uint32_t b) {
            const float x = atlas.templates[a].f.log_aspect, y = atlas.templates[b].f.log_aspect;
            return x != y ? x < y : a < b;
        });
        // Structure-of-arrays copy in scan order: the hot loop streams through memory.
        idx_->aspects.resize(n);
        idx_->holes_f.resize(n);
        idx_->ncomp_f.resize(n);
        idx_->coarse_t.resize(n * 16);
        idx_->bmp.resize(n * kCells);
        for (size_t i = 0; i < n; ++i) {
            const GlyphFeature& f = atlas.templates[idx_->by_aspect[i]].f;
            idx_->aspects[i] = f.log_aspect;
            idx_->holes_f[i] = float(f.holes);
            idx_->ncomp_f[i] = float(f.ncomp);
            for (size_t b = 0; b < 16; ++b) idx_->coarse_t[b * n + i] = f.coarse[b];
            std::copy(f.bmp.begin(), f.bmp.end(), idx_->bmp.begin() + ptrdiff_t(i * kCells));
        }
        if (mode == SearchMode::VPTree) build_vptree();
    }

    // Copying a Classifier is cheap: the template index is shared, scratch buffers and the memo are
    // per copy. Use one copy per thread.
    // Exact memo of classify() results keyed on the full feature. Screens repeat the same glyph
    // raster thousands of times, so most lookups after the first frame are hash hits.
    // Not thread-safe: use one Classifier per thread (they can share the Atlas).
    void enable_memo(bool on, size_t max_entries = 1 << 16) { memo_on_ = on, memo_cap_ = max_entries, memo_.clear(); }
    mutable uint64_t memo_hits = 0;

    const Atlas& atlas() const noexcept { return atlas_; }
    const FeatureWeights& weights() const noexcept { return w_; }
    mutable uint64_t distance_evals = 0, bound_evals = 0;

    // Fills up to k (<=16) best distinct characters, ascending distance. Returns count.
    size_t classify(const GlyphFeature& q, Candidate* out, size_t k) const {
        k = std::min<size_t>(k, 8);
        uint64_t h = 0;
        if (memo_on_) {
            h = hash(q, k);
            auto it = memo_.find(h);
            if (it != memo_.end() && it->second.bmp == q.bmp && it->second.holes == q.holes && it->second.ncomp == q.ncomp &&
                it->second.log_aspect == q.log_aspect && it->second.k == k) {
                ++memo_hits;
                std::copy(it->second.c, it->second.c + it->second.n, out);
                return it->second.n;
            }
        }
        TopChars top(k, margin_);
        if (mode_ == SearchMode::VPTree) search_vp(0, q, top);
        else search_sorted(q, top);
        const size_t n = top.size();
        for (size_t i = 0; i < n; ++i) out[i] = top[i];
        if (memo_on_) {
            if (memo_.size() >= memo_cap_) memo_.clear();
            MemoEntry& e = memo_[h];
            e.bmp = q.bmp, e.holes = q.holes, e.ncomp = q.ncomp, e.log_aspect = q.log_aspect, e.k = k, e.n = n;
            std::copy(out, out + n, e.c);
        }
        return n;
    }

private:
    float dist(const GlyphFeature& q, uint32_t t) const noexcept {
        ++distance_evals;
        return distance(q, atlas_.templates[t].f, w_);
    }

    // Filter & refine. Pass 1 computes the coarse lower bound for *every* template with vertical
    // u16 adds over a block-transposed layout (auto-vectorises to 16 lanes); pass 2 refines
    // best-first in bound order and stops once the bound exceeds the admission threshold tau.
    void search_sorted(const GlyphFeature& q, TopChars& top) const {
        const size_t n = idx_->aspects.size();
        if (n == 0) return;
        lb_.assign(n, 0);
        uint16_t* __restrict lb = lb_.data();
        for (int b = 0; b < 16; ++b) {
            const uint16_t qb = q.coarse[size_t(b)];
            const uint16_t* __restrict col = &idx_->coarse_t[size_t(b) * n];
            // |a-b| as max-min keeps the loop in 16-bit lanes (sum <= 16*4080 = 65280 fits u16).
            for (size_t i = 0; i < n; ++i) lb[i] = uint16_t(lb[i] + uint16_t(std::max(qb, col[i]) - std::min(qb, col[i])));
        }
        bound_.resize(n);
        float* __restrict bd = bound_.data();
        const float qa = q.log_aspect, wa = w_.aspect, wh = w_.holes, wc = w_.ncomp;
        const float qh = float(q.holes), qc = float(q.ncomp);
        const float* __restrict asp = idx_->aspects.data();
        const float* __restrict hol = idx_->holes_f.data();
        const float* __restrict ncp = idx_->ncomp_f.data();
        float m = std::numeric_limits<float>::infinity();
        for (size_t i = 0; i < n; ++i) {
            bd[i] = float(lb[i]) * (1.f / 255.f) + wa * std::fabs(qa - asp[i]) + wh * std::fabs(qh - hol[i]) +
                    wc * std::fabs(qc - ncp[i]);
            m = std::min(m, bd[i]);
        }
        bound_evals += n;

        // Pass A: refine the near-best band to establish tau. Pass B: everything else under tau.
        // Result is exact regardless of order (tau only ever shrinks); order is index-ascending.
        const float band = m + 6.f;
        for (size_t i = 0; i < n; ++i)
            if (bd[i] <= band) refine(q, i, top);
        for (size_t i = 0; i < n; ++i)
            if (bd[i] > band && bd[i] <= top.tau()) refine(q, i, top);
    }

    void refine(const GlyphFeature& q, size_t i, TopChars& top) const {
        ++distance_evals;
        const float extra = w_.aspect * std::fabs(q.log_aspect - idx_->aspects[i]) + w_.holes * std::fabs(float(q.holes) - idx_->holes_f[i]) +
                            w_.ncomp * std::fabs(float(q.ncomp) - idx_->ncomp_f[i]);
        const float tau = top.tau();
        if (extra > tau) return;
        // Early abandon: stop the SAD once it alone exceeds the admission threshold.
        const float room = (tau - extra) * 255.f;
        const uint32_t limit = room >= 4.0e9f ? UINT32_MAX : uint32_t(room);
        const uint32_t sad = sad256_abandon(q.bmp.data(), &idx_->bmp[i * kCells], limit);
        if (sad > limit) return;
        const float d = float(sad) * (1.f / 255.f) + extra;
        const uint32_t t = idx_->by_aspect[i];
        top.offer(atlas_.templates[t].ch, d, t);
    }

    static uint64_t hash(const GlyphFeature& q, size_t k) noexcept {
        uint64_t h = 1469598103934665603ull;  // FNV-1a
        auto mix = [&](uint8_t b) { h = (h ^ b) * 1099511628211ull; };
        for (uint8_t b : q.bmp) mix(b);
        uint32_t la;
        std::memcpy(&la, &q.log_aspect, 4);
        for (int i = 0; i < 4; ++i) mix(uint8_t(la >> (8 * i)));
        mix(uint8_t(q.holes));
        mix(uint8_t(q.ncomp));
        mix(uint8_t(k));
        return h;
    }

    struct MemoEntry {
        std::array<uint8_t, kCells> bmp;
        float log_aspect;
        int8_t holes, ncomp;
        size_t k, n;
        Candidate c[8];
    };
    mutable std::unordered_map<uint64_t, MemoEntry> memo_;
    bool memo_on_ = false;
    size_t memo_cap_ = 1 << 16;

    // ---------------------------------------------------------------- VP-tree
    struct Node {
        uint32_t item;
        float mu = 0;
        int32_t inner = -1, outer = -1;
    };
    // Immutable search data, shared by copies of this Classifier (one copy per thread).
    struct Index {
        std::vector<uint32_t> by_aspect;
        std::vector<float> aspects, holes_f, ncomp_f;
        std::vector<uint16_t> coarse_t;  // [block][template]
        std::vector<uint8_t> bmp;
        std::vector<Node> nodes;
    };
    std::shared_ptr<Index> idx_ = std::make_shared<Index>();

    void build_vptree() {
        std::vector<uint32_t> items(atlas_.templates.size());
        std::iota(items.begin(), items.end(), 0u);
        idx_->nodes.reserve(items.size());
        std::vector<float> d(items.size());
        build(items, 0, items.size(), d);
    }

    int32_t build(std::vector<uint32_t>& items, size_t lo, size_t hi, std::vector<float>& d) {
        if (lo >= hi) return -1;
        const int32_t id = int32_t(idx_->nodes.size());
        idx_->nodes.push_back(Node{items[lo]});
        if (hi - lo == 1) return id;
        // Deterministic vantage choice: the item farthest from the range's first element.
        const GlyphFeature& f0 = atlas_.templates[items[lo]].f;
        size_t vp_i = lo;
        float fd = -1;
        for (size_t i = lo; i < hi; ++i) {
            const float v = distance(f0, atlas_.templates[items[i]].f, w_);
            if (v > fd) fd = v, vp_i = i;
        }
        std::swap(items[lo], items[vp_i]);
        idx_->nodes[size_t(id)].item = items[lo];
        const GlyphFeature& vp = atlas_.templates[items[lo]].f;
        for (size_t i = lo + 1; i < hi; ++i) d[items[i]] = distance(vp, atlas_.templates[items[i]].f, w_);
        const size_t mid = lo + 1 + (hi - lo - 1) / 2;
        std::nth_element(items.begin() + ptrdiff_t(lo + 1), items.begin() + ptrdiff_t(mid), items.begin() + ptrdiff_t(hi),
                         [&](uint32_t a, uint32_t b) { return d[a] != d[b] ? d[a] < d[b] : a < b; });
        idx_->nodes[size_t(id)].mu = d[items[mid]];
        const int32_t in = build(items, lo + 1, mid, d);
        const int32_t out = build(items, mid, hi, d);
        idx_->nodes[size_t(id)].inner = in;
        idx_->nodes[size_t(id)].outer = out;
        return id;
    }

    void search_vp(int32_t id, const GlyphFeature& q, TopChars& top) const {
        if (id < 0) return;
        const Node& n = idx_->nodes[size_t(id)];
        const float d = dist(q, n.item);
        top.offer(atlas_.templates[n.item].ch, d, n.item);
        if (d < n.mu) {
            if (d - top.tau() <= n.mu) search_vp(n.inner, q, top);
            if (d + top.tau() >= n.mu) search_vp(n.outer, q, top);
        } else {
            if (d + top.tau() >= n.mu) search_vp(n.outer, q, top);
            if (d - top.tau() <= n.mu) search_vp(n.inner, q, top);
        }
    }

    const Atlas& atlas_;
    FeatureWeights w_;
    SearchMode mode_;
    float margin_;
    mutable std::vector<uint16_t> lb_;
    mutable std::vector<float> bound_;

};

}  // namespace dks::ocr
