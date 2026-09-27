#pragma once
// ShapeDescriptor: a colour- and polarity-agnostic visual vector for widgets, icons and marks.
//
//   hog[128]   4x4 cells x 8 unsigned gradient orientations on a 24x24 area-resampled luma patch
//              (unsigned = light-on-dark and dark-on-light give the same histogram), L1-normalised
//   ink[64]    8x8 map of |luma - background| (background = patch border median), peak-normalised
//   log_aspect, fill (share of pixels deviating from the background)
//
// Distance: L1(hog) + L1(ink) + W_aspect * |d log2 aspect| + W_fill * |d fill|  (a metric).
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <vector>

#include "../core/image.hpp"

namespace dks::detect {

constexpr int kHog = 128, kInk = 64;

struct ShapeDescriptor {
    std::array<uint8_t, kHog> hog{};
    std::array<uint8_t, kInk> ink{};
    float log_aspect = 0.f;
    float fill = 0.f;
};

struct ShapeWeights {
    float hog = 1.f, ink = 0.6f, aspect = 40.f, fill = 60.f;
};

// Area resample of an arbitrary patch to an n x n float grid.
inline void resample(GrayView g, int n, float* out) {
    for (int i = 0; i < n * n; ++i) out[i] = 0.f;
    const float sx = float(g.width) / float(n), sy = float(g.height) / float(n);
    for (int oy = 0; oy < n; ++oy) {
        const float y0 = float(oy) * sy, y1 = y0 + sy;
        for (int ox = 0; ox < n; ++ox) {
            const float x0 = float(ox) * sx, x1 = x0 + sx;
            float acc = 0.f, wsum = 0.f;
            for (int32_t y = int32_t(y0); y < int32_t(std::ceil(y1)) && y < g.height; ++y) {
                const float wy = std::min(y1, float(y + 1)) - std::max(y0, float(y));
                if (wy <= 0.f) continue;
                for (int32_t x = int32_t(x0); x < int32_t(std::ceil(x1)) && x < g.width; ++x) {
                    const float wx = std::min(x1, float(x + 1)) - std::max(x0, float(x));
                    if (wx <= 0.f) continue;
                    acc += wx * wy * float(g.at(x, y));
                    wsum += wx * wy;
                }
            }
            out[oy * n + ox] = wsum > 0.f ? acc / wsum : 0.f;
        }
    }
}

inline int border_median(GrayView g) {
    std::vector<uint8_t> b;
    for (int32_t x = 0; x < g.width; ++x) b.push_back(g.at(x, 0)), b.push_back(g.at(x, g.height - 1));
    for (int32_t y = 1; y + 1 < g.height; ++y) b.push_back(g.at(0, y)), b.push_back(g.at(g.width - 1, y));
    if (b.empty()) return 0;
    std::nth_element(b.begin(), b.begin() + ptrdiff_t(b.size() / 2), b.end());
    return b[b.size() / 2];
}

inline ShapeDescriptor describe(GrayView g, int fill_delta = 32) {
    ShapeDescriptor d;
    if (g.width < 2 || g.height < 2) return d;
    d.log_aspect = std::log2(float(g.width) / float(g.height));
    const int bg = border_median(g);
    int64_t dev = 0;
    for (int32_t y = 0; y < g.height; ++y)
        for (int32_t x = 0; x < g.width; ++x) dev += std::abs(int(g.at(x, y)) - bg) > fill_delta;
    d.fill = float(dev) / float(g.width * g.height);

    constexpr int N = 24;
    float p[N * N];
    resample(g, N, p);

    // HOG-lite: central differences, unsigned orientation (0..180 deg) into 8 bins, 4x4 cells of 6x6.
    float hog[kHog] = {};
    for (int y = 0; y < N; ++y)
        for (int x = 0; x < N; ++x) {
            const float gx = p[y * N + std::min(N - 1, x + 1)] - p[y * N + std::max(0, x - 1)];
            const float gy = p[std::min(N - 1, y + 1) * N + x] - p[std::max(0, y - 1) * N + x];
            const float mag = std::sqrt(gx * gx + gy * gy);
            if (mag <= 0.f) continue;
            float ang = std::atan2(gy, gx);
            if (ang < 0) ang += 3.14159265f;
            const int bin = std::min(7, int(ang / 3.14159265f * 8.f));
            hog[((y / 6) * 4 + (x / 6)) * 8 + bin] += mag;
        }
    float hs = 0.f;
    for (float v : hog) hs += v;
    for (int i = 0; i < kHog; ++i) d.hog[size_t(i)] = uint8_t(hs > 0 ? std::min(255.f, hog[i] / hs * 255.f * 16.f) : 0.f);

    // Ink map: |luma - background| on an 8x8 grid, peak-normalised.
    float ink[kInk];
    for (int cy = 0; cy < 8; ++cy)
        for (int cx = 0; cx < 8; ++cx) {
            float s = 0.f;
            for (int y = cy * 3; y < cy * 3 + 3; ++y)
                for (int x = cx * 3; x < cx * 3 + 3; ++x) s += std::fabs(p[y * N + x] - float(bg));
            ink[cy * 8 + cx] = s / 9.f;
        }
    float mx = 0.f;
    for (float v : ink) mx = std::max(mx, v);
    for (int i = 0; i < kInk; ++i) d.ink[size_t(i)] = uint8_t(mx > 0 ? ink[i] / mx * 255.f : 0.f);
    return d;
}

inline float distance(const ShapeDescriptor& a, const ShapeDescriptor& b, const ShapeWeights& w = {}) {
    uint32_t h = 0, k = 0;
    for (int i = 0; i < kHog; ++i) h += uint32_t(std::abs(int(a.hog[size_t(i)]) - int(b.hog[size_t(i)])));
    for (int i = 0; i < kInk; ++i) k += uint32_t(std::abs(int(a.ink[size_t(i)]) - int(b.ink[size_t(i)])));
    return w.hog * float(h) / 255.f + w.ink * float(k) / 255.f + w.aspect * std::fabs(a.log_aspect - b.log_aspect) +
           w.fill * std::fabs(a.fill - b.fill);
}

}  // namespace dks::detect
