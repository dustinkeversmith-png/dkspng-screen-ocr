#pragma once
// Axis-aligned integer rectangles. Half-open: [x, x+w) x [y, y+h).
#include <algorithm>
#include <cstdint>

namespace dks {

struct Rect {
    int32_t x = 0, y = 0, w = 0, h = 0;

    constexpr int32_t right() const noexcept { return x + w; }
    constexpr int32_t bottom() const noexcept { return y + h; }
    constexpr int64_t area() const noexcept { return int64_t(w) * h; }
    constexpr bool empty() const noexcept { return w <= 0 || h <= 0; }
    constexpr int32_t cx2() const noexcept { return 2 * x + w; }  // 2x centre, stays integral
    constexpr int32_t cy2() const noexcept { return 2 * y + h; }

    static constexpr Rect from_ltrb(int32_t l, int32_t t, int32_t r, int32_t b) noexcept {
        return Rect{l, t, r - l, b - t};
    }

    constexpr bool contains(const Rect& o) const noexcept {
        return x <= o.x && y <= o.y && right() >= o.right() && bottom() >= o.bottom();
    }
    constexpr bool contains_point(int32_t px, int32_t py) const noexcept {
        return px >= x && py >= y && px < right() && py < bottom();
    }

    Rect intersect(const Rect& o) const noexcept {
        const int32_t l = std::max(x, o.x), t = std::max(y, o.y);
        const int32_t r = std::min(right(), o.right()), b = std::min(bottom(), o.bottom());
        return (r > l && b > t) ? from_ltrb(l, t, r, b) : Rect{};
    }
    Rect unite(const Rect& o) const noexcept {
        if (empty()) return o;
        if (o.empty()) return *this;
        return from_ltrb(std::min(x, o.x), std::min(y, o.y), std::max(right(), o.right()),
                         std::max(bottom(), o.bottom()));
    }
    Rect inflate(int32_t d) const noexcept { return Rect{x - d, y - d, w + 2 * d, h + 2 * d}; }
    Rect clip(int32_t W, int32_t H) const noexcept { return intersect(Rect{0, 0, W, H}); }

    int64_t inter_area(const Rect& o) const noexcept { return intersect(o).area(); }

    float iou(const Rect& o) const noexcept {
        const int64_t i = inter_area(o);
        if (i == 0) return 0.0f;
        return float(double(i) / double(area() + o.area() - i));
    }

    // Overlap of the projections onto each axis.
    int32_t x_overlap(const Rect& o) const noexcept { return std::min(right(), o.right()) - std::max(x, o.x); }
    int32_t y_overlap(const Rect& o) const noexcept { return std::min(bottom(), o.bottom()) - std::max(y, o.y); }

    constexpr bool operator==(const Rect& o) const noexcept {
        return x == o.x && y == o.y && w == o.w && h == o.h;
    }
};

}  // namespace dks
