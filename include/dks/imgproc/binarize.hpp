#pragma once
// Binarisation primitives.
//   * edge_mask   : polarity-free structure map for whole-screen segmentation (text, borders, icons).
//   * otsu / ink  : polarity-aware ink extraction for a single text crop (recognition stage).
#include <array>
#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <vector>

#include "../core/image.hpp"

namespace dks {

// ----------------------------------------------------------------------------------------- edges
// A pixel is "structure" if any colour channel differs from its right or bottom neighbour by more
// than `threshold`. Flat UI regions vanish, glyph strokes / borders / icon detail survive, and both
// dark-on-light and light-on-dark content produce the same map (no polarity decision needed).
// The edge between p and p+1 is attributed to *both* pixels so 1px strokes stay 8-connected.
inline void edge_mask(const ColorView& src, ImageView<uint8_t> dst, int threshold) noexcept {
    const int bpp = bytes_per_pixel(src.format);
    const int32_t W = src.width, H = src.height;
    for (int32_t y = 0; y < H; ++y) {
        uint8_t* d = dst.row(y);
        for (int32_t x = 0; x < W; ++x) d[x] = 0;
    }
    const int nch = bpp >= 3 ? 3 : 1;
    for (int32_t y = 0; y < H; ++y) {
        const uint8_t* s = src.row(y);
        const uint8_t* sd = (y + 1 < H) ? src.row(y + 1) : nullptr;
        uint8_t* d = dst.row(y);
        uint8_t* dd = (y + 1 < H) ? dst.row(y + 1) : nullptr;
        for (int32_t x = 0; x < W; ++x) {
            const uint8_t* p = s + x * bpp;
            int mr = 0, md = 0;
            for (int c = 0; c < nch; ++c) {
                if (x + 1 < W) { const int v = std::abs(int(p[c]) - int(p[bpp + c])); mr = v > mr ? v : mr; }
                if (sd) { const int v = std::abs(int(p[c]) - int(sd[x * bpp + c])); md = v > md ? v : md; }
            }
            if (mr > threshold) { d[x] = 1; d[x + 1] = 1; }
            if (md > threshold) { d[x] = 1; dd[x] = 1; }
        }
    }
}

// ----------------------------------------------------------------------------------------- otsu
using Histogram = std::array<uint32_t, 256>;

inline Histogram histogram(GrayView g) noexcept {
    Histogram h{};
    for (int32_t y = 0; y < g.height; ++y) {
        const uint8_t* r = g.row(y);
        for (int32_t x = 0; x < g.width; ++x) ++h[r[x]];
    }
    return h;
}

// Returns t such that class0 = [0, t], class1 = (t, 255].
inline int otsu_threshold(const Histogram& h) noexcept {
    uint64_t total = 0, sum = 0;
    for (int i = 0; i < 256; ++i) total += h[i], sum += uint64_t(i) * h[i];
    if (total == 0) return 127;
    uint64_t w0 = 0, sum0 = 0;
    double best = -1.0;
    int bt = 127;
    for (int t = 0; t < 255; ++t) {
        w0 += h[t];
        sum0 += uint64_t(t) * h[t];
        if (w0 == 0) continue;
        const uint64_t w1 = total - w0;
        if (w1 == 0) break;
        const double m0 = double(sum0) / double(w0), m1 = double(sum - sum0) / double(w1);
        const double between = double(w0) * double(w1) * (m0 - m1) * (m0 - m1);
        if (between > best) best = between, bt = t;
    }
    return bt;
}

// Ink extraction for a text crop. Decides polarity from the crop border (background dominates the
// border of a text crop), then produces:
//   mask : 1 = ink
//   ink  : contrast-normalised ink intensity, 0 = background level, 255 = ink level (soft, keeps AA)
struct InkResult {
    Gray8 mask, ink;
    int threshold = 127;
    bool dark_text = true;
    int bg_level = 255, fg_level = 0;
    int contrast() const noexcept { return std::abs(bg_level - fg_level); }
};

struct InkParams {
    // Strip outer rows/columns that are >= 95% one Otsu class before re-running Otsu and the border
    // vote on the inner region (removes uniform frame bands from the histogram).
    bool trim_margins = false;
    // Sauvola local threshold on the level-normalised ink (T = m * (1 + k * (s/128 - 1))), window
    // = window_h * crop height, AND-ed with a weak coverage floor.
    bool sauvola = false;
    float sauvola_k = 0.2f;
    float window_h = 1.2f;
    // Shadowed / outlined text (desktop icon labels, banner text): three luminance populations
    // (shadow < background < text or the reverse). When the border median sits between two
    // strong populations, ink = the clearly sharper side (shadows / glows are blurred), else the side
    // with the larger contrast from the background (crisp outlined banner text).
    bool shadow_mode = true;
    int shadow_min_dev = 40;          // both sides must deviate at least this much (p2 / p98 vs border median)
    double shadow_sharp_ratio = 2.0;  // the sharper side must be this much sharper to override contrast
};

inline InkResult extract_ink(GrayView g, const InkParams& IP = {}) {
    InkResult r;
    const int32_t W = g.width, H = g.height;
    r.mask.reset(W, H, 0);
    r.ink.reset(W, H, 0);
    if (W <= 0 || H <= 0) return r;
    Histogram hist = histogram(g);
    int t = otsu_threshold(hist);

    Rect in{0, 0, W, H};
    if (IP.trim_margins) {
        auto uniform_row = [&](int32_t y) {
            int32_t n = 0;
            for (int32_t x = in.x; x < in.right(); ++x) n += g.at(x, y) > t;
            return n * 20 >= in.w * 19 || n * 20 <= in.w;
        };
        auto uniform_col = [&](int32_t x) {
            int32_t n = 0;
            for (int32_t y = in.y; y < in.bottom(); ++y) n += g.at(x, y) > t;
            return n * 20 >= in.h * 19 || n * 20 <= in.h;
        };
        while (in.h > 4 && uniform_row(in.y)) ++in.y, --in.h;
        while (in.h > 4 && uniform_row(in.bottom() - 1)) --in.h;
        while (in.w > 4 && uniform_col(in.x)) ++in.x, --in.w;
        while (in.w > 4 && uniform_col(in.right() - 1)) --in.w;
        if (!(in == Rect{0, 0, W, H})) {
            hist = histogram(g.sub(in));
            t = otsu_threshold(hist);
        }
    }
    r.threshold = t;

    // Border vote: which Otsu class owns the (inner) crop frame?
    uint64_t above = 0, below = 0;
    auto vote = [&](int32_t x, int32_t y) { (g.at(x, y) > t ? above : below) += 1; };
    for (int32_t x = in.x; x < in.right(); ++x) { vote(x, in.y); vote(x, in.bottom() - 1); }
    for (int32_t y = in.y + 1; y + 1 < in.bottom(); ++y) { vote(in.x, y); vote(in.right() - 1, y); }
    r.dark_text = above >= below;

    // Ambiguous vote (text touching the crop frame, common for tight detector boxes): fall back to
    // stroke geometry. Ink forms short horizontal runs (stroke widths); background forms long ones.
    const uint64_t votes = above + below;
    if (votes > 0 && std::max(above, below) * 4 < votes * 3) {
        uint64_t runs[2] = {0, 0}, len[2] = {0, 0};  // [0] = class <= t, [1] = class > t
        for (int32_t y = 0; y < H; ++y) {
            const uint8_t* s = g.row(y);
            int32_t x = 0;
            while (x < W) {
                const int c = s[x] > t ? 1 : 0;
                int32_t e = x + 1;
                while (e < W && (s[e] > t ? 1 : 0) == c) ++e;
                if (x > 0 && e < W) ++runs[c], len[c] += uint64_t(e - x);  // interior runs only
                x = e;
            }
        }
        if (runs[0] && runs[1]) {
            // mean run length comparison: len0/runs0 < len1/runs1  <=>  dark class is the thin one
            r.dark_text = len[0] * runs[1] < len[1] * runs[0];
        }
    }

    if (IP.shadow_mode) {
        std::vector<uint8_t> border;
        for (int32_t x = 0; x < W; ++x) border.push_back(g.at(x, 0)), border.push_back(g.at(x, H - 1));
        for (int32_t y = 1; y + 1 < H; ++y) border.push_back(g.at(0, y)), border.push_back(g.at(W - 1, y));
        std::nth_element(border.begin(), border.begin() + ptrdiff_t(border.size() / 2), border.end());
        const int bgm = border[border.size() / 2];
        auto pct = [&](int q) {
            uint64_t n = 0, acc = 0;
            for (int i = 0; i < 256; ++i) n += hist[i];
            for (int i = 0; i < 256; ++i)
                if ((acc += hist[i]) * 100 >= uint64_t(q) * n) return i;
            return 255;
        };
        const int lo = pct(2), hi = pct(98);
        const int dark_dev = bgm - lo, light_dev = hi - bgm;
        if (std::min(dark_dev, light_dev) >= IP.shadow_min_dev) {
            // Pick the text side. A drop shadow / glow is *blurred*: few of its pixels reach full
            // coverage. Text (and crisp outlines) is sharp. The sharper side wins when clearly
            // sharper; otherwise the side with the larger contrast (outlined banner text).
            auto sharpness = [&](bool dark) {
                const int dev = dark ? dark_dev : light_dev;
                uint64_t part = 0, solid = 0;
                for (int32_t y = 0; y < H; ++y) {
                    const uint8_t* src = g.row(y);
                    for (int32_t x = 0; x < W; ++x) {
                        const int d = dark ? bgm - src[x] : src[x] - bgm;
                        if (4 * d >= dev) ++part;
                        if (4 * d >= 3 * dev) ++solid;
                    }
                }
                return part ? double(solid) / double(part) : 0.0;
            };
            const double sd = sharpness(true), sl = sharpness(false), k = IP.shadow_sharp_ratio;
            r.dark_text = sd > sl * k ? true : (sl > sd * k ? false : dark_dev > light_dev);
            r.bg_level = bgm;
            r.fg_level = r.dark_text ? lo : hi;
            const int span = std::max(1, std::abs(r.bg_level - r.fg_level));
            for (int32_t y = 0; y < H; ++y) {
                const uint8_t* src = g.row(y);
                uint8_t* m = r.mask.row(y);
                uint8_t* k = r.ink.row(y);
                for (int32_t x = 0; x < W; ++x) {
                    int s255 = (r.dark_text ? (r.bg_level - src[x]) : (src[x] - r.bg_level)) * 255 / span;
                    s255 = s255 < 0 ? 0 : (s255 > 255 ? 255 : s255);  // the opposite side (shadow) -> 0
                    k[x] = uint8_t(s255);
                    m[x] = s255 >= 128 ? 1 : 0;
                }
            }
            r.threshold = (r.bg_level + r.fg_level) / 2;
            return r;
        }
    }

    // Robust class levels: median of each class.
    auto class_median = [&](int lo, int hi) {
        uint64_t n = 0;
        for (int i = lo; i <= hi; ++i) n += hist[i];
        uint64_t acc = 0;
        for (int i = lo; i <= hi; ++i) { acc += hist[i]; if (2 * acc >= n) return i; }
        return (lo + hi) / 2;
    };
    const int lo_level = class_median(0, t), hi_level = class_median(t + 1, 255);
    r.bg_level = r.dark_text ? hi_level : lo_level;
    r.fg_level = r.dark_text ? lo_level : hi_level;

    // Ink = >= 50% coverage between the two levels. Unlike the raw Otsu cut this is a property of
    // the glyph raster alone, so an isolated template glyph and the same glyph inside a busy
    // coloured line crop get the same mask (Otsu drifts with the crop's histogram, which moves
    // antialiased edge columns in/out of the mask and breaks small-text matching).
    const int span = std::max(1, std::abs(r.bg_level - r.fg_level));
    for (int32_t y = 0; y < H; ++y) {
        const uint8_t* s = g.row(y);
        uint8_t* m = r.mask.row(y);
        uint8_t* k = r.ink.row(y);
        for (int32_t x = 0; x < W; ++x) {
            const int v = s[x];
            int s255 = (r.dark_text ? (r.bg_level - v) : (v - r.bg_level)) * 255 / span;
            s255 = s255 < 0 ? 0 : (s255 > 255 ? 255 : s255);
            k[x] = uint8_t(s255);
            m[x] = s255 >= 128 ? 1 : 0;
        }
    }
    if (IP.sauvola) {
        // Integral images over the inverted ink (ink dark, background bright), 64-bit sums.
        const int32_t win = std::max(3, int32_t(IP.window_h * float(H))) | 1;
        std::vector<uint64_t> S1(size_t(W + 1) * size_t(H + 1), 0), S2(S1.size(), 0);
        for (int32_t y = 0; y < H; ++y) {
            uint64_t a = 0, b = 0;
            for (int32_t x = 0; x < W; ++x) {
                const uint64_t v = 255u - r.ink.at(x, y);
                a += v, b += v * v;
                S1[size_t(y + 1) * size_t(W + 1) + size_t(x + 1)] = S1[size_t(y) * size_t(W + 1) + size_t(x + 1)] + a;
                S2[size_t(y + 1) * size_t(W + 1) + size_t(x + 1)] = S2[size_t(y) * size_t(W + 1) + size_t(x + 1)] + b;
            }
        }
        auto box = [&](const std::vector<uint64_t>& S, int32_t x0, int32_t y0, int32_t x1, int32_t y1) {
            const size_t w1 = size_t(W + 1);
            return double(S[size_t(y1) * w1 + size_t(x1)] - S[size_t(y0) * w1 + size_t(x1)] - S[size_t(y1) * w1 + size_t(x0)] +
                          S[size_t(y0) * w1 + size_t(x0)]);
        };
        for (int32_t y = 0; y < H; ++y) {
            const int32_t y0 = std::max(0, y - win / 2), y1 = std::min(H, y + win / 2 + 1);
            for (int32_t x = 0; x < W; ++x) {
                const int32_t x0 = std::max(0, x - win / 2), x1 = std::min(W, x + win / 2 + 1);
                const double n = double((x1 - x0) * (y1 - y0));
                const double mean = box(S1, x0, y0, x1, y1) / n;
                const double sd = std::sqrt(std::max(0.0, box(S2, x0, y0, x1, y1) / n - mean * mean));
                const double T = mean * (1.0 + double(IP.sauvola_k) * (sd / 128.0 - 1.0));
                const int v = 255 - r.ink.at(x, y);
                r.mask.at(x, y) = (double(v) < T && r.ink.at(x, y) >= 64) ? 1 : 0;
            }
        }
    }
    return r;
}

}  // namespace dks
