#pragma once
// Run-based two-row connected-component labelling (union-find).
//
// Only two rows of runs are live while scanning; no W*H label image is ever allocated. Runs are
// kept afterwards (grouped per component) so a component's exact mask can be re-rasterised on
// demand, e.g. to cut out one glyph without its neighbours.
#include <algorithm>
#include <cstdint>
#include <vector>

#include "../core/image.hpp"

namespace dks {

class DisjointSet {
public:
    void clear() { parent_.clear(); }
    uint32_t make() {
        parent_.push_back(uint32_t(parent_.size()));
        return parent_.back();
    }
    uint32_t find(uint32_t i) noexcept {
        uint32_t root = i;
        while (root != parent_[root]) root = parent_[root];
        while (i != root) {  // path compression
            const uint32_t next = parent_[i];
            parent_[i] = root;
            i = next;
        }
        return root;
    }
    // Union by smaller index keeps roots stable & deterministic (root = earliest-seen run).
    void unite(uint32_t a, uint32_t b) noexcept {
        a = find(a), b = find(b);
        if (a == b) return;
        if (a < b) parent_[b] = a; else parent_[a] = b;
    }
    size_t size() const noexcept { return parent_.size(); }

private:
    std::vector<uint32_t> parent_;
};

struct Run {
    int32_t y, x0, x1;  // [x0, x1)
    uint32_t label;
};

struct Component {
    Rect bbox;
    int32_t area = 0;        // ink pixels
    uint32_t run_begin = 0;  // into LabelResult::runs (grouped by component)
    uint32_t run_end = 0;
    float fill() const noexcept { return bbox.area() ? float(area) / float(bbox.area()) : 0.f; }
};

enum class Connectivity : uint8_t { Four = 4, Eight = 8 };

struct LabelResult {
    std::vector<Component> components;
    std::vector<Run> runs;  // sorted by component, then (y, x0)
    int32_t width = 0, height = 0;

    // Paint component `c` into `dst` (same coordinate frame as the labelled image, offset by -origin).
    void rasterize(uint32_t c, ImageView<uint8_t> dst, int32_t ox = 0, int32_t oy = 0, uint8_t value = 1) const {
        const Component& k = components[c];
        for (uint32_t i = k.run_begin; i < k.run_end; ++i) {
            const Run& r = runs[i];
            const int32_t y = r.y - oy;
            if (y < 0 || y >= dst.height) continue;
            uint8_t* row = dst.row(y);
            for (int32_t x = std::max(0, r.x0 - ox); x < std::min(dst.width, r.x1 - ox); ++x) row[x] = value;
        }
    }
};

// Label all non-zero pixels of `mask`.
inline LabelResult label_components(GrayView mask, Connectivity conn = Connectivity::Eight) {
    LabelResult out;
    out.width = mask.width;
    out.height = mask.height;
    std::vector<Run>& runs = out.runs;
    runs.reserve(size_t(mask.height) * 4);
    DisjointSet ds;
    const int32_t slack = conn == Connectivity::Eight ? 1 : 0;

    size_t prev_begin = 0, prev_end = 0;
    for (int32_t y = 0; y < mask.height; ++y) {
        const uint8_t* row = mask.row(y);
        const size_t cur_begin = runs.size();
        int32_t x = 0;
        const int32_t W = mask.width;
        while (x < W) {
            while (x < W && !row[x]) ++x;
            if (x >= W) break;
            const int32_t x0 = x;
            while (x < W && row[x]) ++x;
            runs.push_back(Run{y, x0, x, UINT32_MAX});
        }
        const size_t cur_end = runs.size();

        // Two-pointer merge against the previous row's runs.
        size_t p = prev_begin;
        for (size_t c = cur_begin; c < cur_end; ++c) {
            Run& cr = runs[c];
            while (p < prev_end && runs[p].x1 + slack <= cr.x0) ++p;
            for (size_t q = p; q < prev_end && runs[q].x0 < cr.x1 + slack; ++q) {
                if (cr.label == UINT32_MAX) cr.label = runs[q].label;
                else ds.unite(cr.label, runs[q].label);
            }
            if (cr.label == UINT32_MAX) cr.label = ds.make();
        }
        prev_begin = cur_begin;
        prev_end = cur_end;
    }

    // Resolve provisional labels -> dense component ids (in order of first appearance).
    std::vector<uint32_t> dense(ds.size(), UINT32_MAX);
    uint32_t n = 0;
    for (Run& r : runs) {
        const uint32_t root = ds.find(r.label);
        if (dense[root] == UINT32_MAX) dense[root] = n++;
        r.label = dense[root];
    }

    // Stats + counting sort of runs by component.
    auto& comps = out.components;
    comps.assign(n, Component{});
    std::vector<int32_t> l(n, INT32_MAX), t(n, INT32_MAX), rr(n, INT32_MIN), b(n, INT32_MIN);
    std::vector<uint32_t> count(n + 1, 0);
    for (const Run& r : runs) {
        const uint32_t k = r.label;
        comps[k].area += r.x1 - r.x0;
        l[k] = std::min(l[k], r.x0);
        rr[k] = std::max(rr[k], r.x1);
        t[k] = std::min(t[k], r.y);
        b[k] = std::max(b[k], r.y + 1);
        ++count[k + 1];
    }
    for (uint32_t k = 0; k < n; ++k) count[k + 1] += count[k];
    for (uint32_t k = 0; k < n; ++k) {
        comps[k].bbox = Rect::from_ltrb(l[k], t[k], rr[k], b[k]);
        comps[k].run_begin = count[k];
        comps[k].run_end = count[k + 1];
    }
    std::vector<Run> sorted(runs.size());
    for (const Run& r : runs) sorted[count[r.label]++] = r;  // stable: preserves (y, x0) order
    runs.swap(sorted);
    return out;
}

}  // namespace dks
