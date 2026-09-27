#pragma once
// Glyph descriptor: aspect-preserving 16x16 area-resampled ink bitmap + log aspect + topology.
//
// Distance (a true metric, so metric trees apply):
//     d(a,b) = L1(bmp_a, bmp_b)/255 + W_aspect*|log2 aspect_a - log2 aspect_b|
//              + W_holes*|holes_a - holes_b| + W_ncomp*|components_a - components_b|
// Holes / components come from the Euler number chi = C - H (Gray's 2x2 quad counts) plus a CCL
// component count, so the topological invariant is split into its two informative parts.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>

#include "../core/image.hpp"
#include "../segment/ccl.hpp"
#include "simd_metric.hpp"

namespace dks::ocr {

constexpr int kGrid = 16;
constexpr int kCells = kGrid * kGrid;

struct GlyphFeature {
    alignas(32) std::array<uint8_t, kCells> bmp{};
    std::array<uint16_t, 16> coarse{};  // 4x4 block sums of bmp: cheap exact lower bound on L1
    float log_aspect = 0.f;  // log2(w / h)
    int8_t euler = 1;        // chi = components - holes (8-connected foreground)
    int8_t ncomp = 1;        // connected components (i, j, : = 2; % = 3; merged neighbours > 1)
    int8_t holes = 0;        // ncomp - chi   (o, a, e = 1; B, 8 = 2)
    int16_t w = 0, h = 0;    // source size in px
};

struct FeatureWeights {
    float aspect = 14.f;  // per unit of log2 aspect
    float holes = 6.f;    // per hole of difference
    float ncomp = 10.f;   // per connected component of difference
};

// Non-bitmap part of the metric (shared by distance(), lower_bound() and the classifier scan).
inline float scalar_terms(const GlyphFeature& a, const GlyphFeature& b, const FeatureWeights& w) noexcept {
    return w.aspect * std::fabs(a.log_aspect - b.log_aspect) + w.holes * float(std::abs(a.holes - b.holes)) +
           w.ncomp * float(std::abs(a.ncomp - b.ncomp));
}

inline uint32_t sad256(const uint8_t* __restrict a, const uint8_t* __restrict b) noexcept { return sad256_simd(a, b); }

inline float distance(const GlyphFeature& a, const GlyphFeature& b, const FeatureWeights& w = {}) noexcept {
    return float(sad256(a.bmp.data(), b.bmp.data())) * (1.f / 255.f) + scalar_terms(a, b, w);
}

// Exact lower bound of distance(): |sum a - sum b| <= sum |a - b| per 4x4 block.
inline float lower_bound(const GlyphFeature& a, const GlyphFeature& b, const FeatureWeights& w = {}) noexcept {
    uint32_t s = 0;
    for (int i = 0; i < 16; ++i) s += uint32_t(std::abs(int(a.coarse[size_t(i)]) - int(b.coarse[size_t(i)])));
    return float(s) * (1.f / 255.f) + scalar_terms(a, b, w);
}

inline bool& smooth_features() noexcept {
    static bool on = true;
    return on;
}

// Euler number of a binary mask (8-connectivity foreground) from 2x2 quad counts (Gray 1971):
//   chi_8 = (C1 - C3 - 2*CD) / 4, with a zero frame so boundary quads are counted.
inline int euler_number(GrayView m) noexcept {
    int c1 = 0, c3 = 0, cd = 0;
    auto px = [&](int32_t x, int32_t y) -> int {
        return (x >= 0 && y >= 0 && x < m.width && y < m.height && m.at(x, y)) ? 1 : 0;
    };
    for (int32_t y = -1; y < m.height; ++y) {
        for (int32_t x = -1; x < m.width; ++x) {
            const int a = px(x, y), b = px(x + 1, y), c = px(x, y + 1), d = px(x + 1, y + 1);
            const int s = a + b + c + d;
            if (s == 1) ++c1;
            else if (s == 3) ++c3;
            else if (s == 2 && a == d) ++cd;  // diagonal pair (a&d) or (b&c)
        }
    }
    return (c1 - c3 - 2 * cd) / 4;
}

// Build the descriptor from a glyph-local soft ink image (0 = background) and its binary mask,
// both exactly the glyph's bounding box in size.
inline GlyphFeature make_feature(GrayView ink, GrayView mask) {
    GlyphFeature f;
    const int32_t w = ink.width, h = ink.height;
    f.w = int16_t(w), f.h = int16_t(h);
    if (w <= 0 || h <= 0) return f;
    f.log_aspect = std::log2(float(w) / float(h));
    f.euler = int8_t(std::clamp(euler_number(mask), -3, 3));
    f.ncomp = int8_t(std::min<size_t>(4, label_components(mask, Connectivity::Eight).components.size()));
    f.holes = int8_t(std::clamp(int(f.ncomp) - int(f.euler), 0, 3));

    // Area splatting handles both up- and down-sampling exactly.
    const float s = float(kGrid) / float(std::max(w, h));
    const float ox = (float(kGrid) - float(w) * s) * 0.5f, oy = (float(kGrid) - float(h) * s) * 0.5f;
    std::array<float, kCells> acc{};
    for (int32_t y = 0; y < h; ++y) {
        const float y0 = oy + float(y) * s, y1 = y0 + s;
        const int gy0 = std::max(0, int(std::floor(y0))), gy1 = std::min(kGrid - 1, int(std::ceil(y1)) - 1);
        const uint8_t* row = ink.row(y);
        for (int32_t x = 0; x < w; ++x) {
            const float v = float(row[x]);
            if (v == 0.f) continue;
            const float x0 = ox + float(x) * s, x1 = x0 + s;
            const int gx0 = std::max(0, int(std::floor(x0))), gx1 = std::min(kGrid - 1, int(std::ceil(x1)) - 1);
            for (int gy = gy0; gy <= gy1; ++gy) {
                const float cy = std::min(y1, float(gy + 1)) - std::max(y0, float(gy));
                if (cy <= 0.f) continue;
                for (int gx = gx0; gx <= gx1; ++gx) {
                    const float cx = std::min(x1, float(gx + 1)) - std::max(x0, float(gx));
                    if (cx > 0.f) acc[size_t(gy * kGrid + gx)] += v * cx * cy;
                }
            }
        }
    }
    // Optional [1 2 1]^2 smoothing on the grid: tolerance to 1-cell shifts / stroke weight / blur.
    if (smooth_features()) {
        std::array<float, kCells> t{};
        for (int y = 0; y < kGrid; ++y)
            for (int x = 0; x < kGrid; ++x) {
                float s = 2.f * acc[size_t(y * kGrid + x)];
                s += x > 0 ? acc[size_t(y * kGrid + x - 1)] : 0.f;
                s += x + 1 < kGrid ? acc[size_t(y * kGrid + x + 1)] : 0.f;
                t[size_t(y * kGrid + x)] = s;
            }
        for (int y = 0; y < kGrid; ++y)
            for (int x = 0; x < kGrid; ++x) {
                float s = 2.f * t[size_t(y * kGrid + x)];
                s += y > 0 ? t[size_t((y - 1) * kGrid + x)] : 0.f;
                s += y + 1 < kGrid ? t[size_t((y + 1) * kGrid + x)] : 0.f;
                acc[size_t(y * kGrid + x)] = s;
            }
    }
    // Normalise peak to 255: removes stroke-weight / antialias-contrast differences between renderers.
    float mx = 0.f;
    for (float a : acc) mx = std::max(mx, a);
    const float k = mx > 0.f ? 255.f / mx : 0.f;
    for (int i = 0; i < kCells; ++i) f.bmp[size_t(i)] = uint8_t(std::min(255.f, acc[size_t(i)] * k + 0.5f));
    for (int i = 0; i < kCells; ++i) f.coarse[size_t(((i / kGrid) / 4) * 4 + (i % kGrid) / 4)] += f.bmp[size_t(i)];
    return f;
}

}  // namespace dks::ocr
