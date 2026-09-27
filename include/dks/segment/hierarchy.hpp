#pragma once
// Containment forest over rectangles: parent(B) = smallest-area rect that fully contains B.
//
// Boxes are inserted largest-first into a uniform grid; a box registers in every cell it covers.
// Any container of B must cover B's top-left pixel, so the query only scans that one cell, whose
// list is already ordered large -> small: the last containing entry is the tightest parent.
// Cost ~ O(N log N + sum of covered cells), independent of pairwise N^2 comparisons.
#include <algorithm>
#include <cstdint>
#include <numeric>
#include <vector>

#include "../core/geometry.hpp"

namespace dks {

struct Forest {
    std::vector<int32_t> parent;  // -1 for roots
    std::vector<int32_t> depth;   // 0 for roots
};

inline Forest build_containment_forest(const std::vector<Rect>& boxes, int32_t cell = 64) {
    const size_t n = boxes.size();
    Forest f;
    f.parent.assign(n, -1);
    f.depth.assign(n, 0);
    if (n == 0) return f;

    int32_t minx = INT32_MAX, miny = INT32_MAX, maxx = INT32_MIN, maxy = INT32_MIN;
    for (const Rect& r : boxes) {
        minx = std::min(minx, r.x), miny = std::min(miny, r.y);
        maxx = std::max(maxx, r.right()), maxy = std::max(maxy, r.bottom());
    }
    const int32_t gw = std::max(1, (maxx - minx + cell - 1) / cell + 1);
    const int32_t gh = std::max(1, (maxy - miny + cell - 1) / cell + 1);
    std::vector<std::vector<uint32_t>> grid(size_t(gw) * gh);
    auto cx = [&](int32_t x) { return std::clamp((x - minx) / cell, 0, gw - 1); };
    auto cy = [&](int32_t y) { return std::clamp((y - miny) / cell, 0, gh - 1); };

    std::vector<uint32_t> order(n);
    std::iota(order.begin(), order.end(), 0u);
    std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
        return boxes[a].area() > boxes[b].area();
    });

    for (uint32_t i : order) {
        const Rect& r = boxes[i];
        const auto& cand = grid[size_t(cy(r.y)) * gw + cx(r.x)];
        for (size_t k = cand.size(); k-- > 0;) {
            const uint32_t j = cand[k];
            if (boxes[j].contains(r)) {
                f.parent[i] = int32_t(j);
                f.depth[i] = f.depth[j] + 1;
                break;
            }
        }
        if (r.empty()) continue;
        const int32_t x0 = cx(r.x), x1 = cx(r.right() - 1), y0 = cy(r.y), y1 = cy(r.bottom() - 1);
        for (int32_t gy = y0; gy <= y1; ++gy)
            for (int32_t gx = x0; gx <= x1; ++gx) grid[size_t(gy) * gw + gx].push_back(i);
    }
    return f;
}

}  // namespace dks
