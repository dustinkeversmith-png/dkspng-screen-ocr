// === C:\Users\Cutie Magic 500\projects\creative\screen-ocr\bench\common.hpp ===
#pragma once
// Shared bench helpers: manifest / GT loading and debug overlays.
#include <chrono>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "dks/core/geometry.hpp"
#include "dks/platform/win32.hpp"

namespace bench {

struct GtBox {
    dks::Rect box;
    std::string label, text;
};

struct Sample {
    std::string image, gt;  // absolute-ish paths
};

inline std::string unescape(const std::string& s) {
    std::string o;
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '\\' && i + 1 < s.size()) {
            const char c = s[++i];
            o.push_back(c == 't' ? '\t' : c == 'n' ? '\n' : c);
        } else o.push_back(s[i]);
    }
    return o;
}

inline std::vector<std::string> split_tabs(const std::string& line) {
    std::vector<std::string> f;
    std::string cur;
    for (char c : line) {
        if (c == '\t') f.push_back(cur), cur.clear();
        else if (c != '\r') cur.push_back(c);
    }
    f.push_back(cur);
    return f;
}

inline std::vector<Sample> load_manifest(const std::string& dir) {
    std::vector<Sample> out;
    std::ifstream in(dir + "/manifest.tsv", std::ios::binary);
    std::string line;
    while (std::getline(in, line)) {
        const auto f = split_tabs(line);
        if (f.size() >= 2) out.push_back({dir + "/" + f[0], dir + "/" + f[1]});
    }
    return out;
}

inline std::vector<GtBox> load_gt(const std::string& path) {
    std::vector<GtBox> out;
    std::ifstream in(path, std::ios::binary);
    std::string line;
    while (std::getline(in, line)) {
        const auto f = split_tabs(line);
        if (f.size() < 5) continue;
        GtBox g;
        g.box = dks::Rect{std::stoi(f[0]), std::stoi(f[1]), std::stoi(f[2]), std::stoi(f[3])};
        g.label = unescape(f[4]);
        g.text = f.size() > 5 ? unescape(f[5]) : "";
        out.push_back(std::move(g));
    }
    return out;
}

struct Timer {
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    double ms() const {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    }
};

inline void draw_rect(dks::win32::Frame& f, const dks::Rect& r, uint8_t R, uint8_t G, uint8_t B, int thick = 1) {
    auto put = [&](int32_t x, int32_t y) {
        if (x < 0 || y < 0 || x >= f.width || y >= f.height) return;
        uint8_t* p = &f.pixels[(size_t(y) * f.width + x) * 4];
        p[0] = B, p[1] = G, p[2] = R, p[3] = 255;
    };
    for (int t = 0; t < thick; ++t) {
        for (int32_t x = r.x - t; x < r.right() + t; ++x) put(x, r.y - t), put(x, r.bottom() - 1 + t);
        for (int32_t y = r.y - t; y < r.bottom() + t; ++y) put(r.x - t, y), put(r.right() - 1 + t, y);
    }
}

inline std::string arg_value(int argc, char** argv, const std::string& key, const std::string& def) {
    for (int i = 1; i + 1 < argc; ++i)
        if (argv[i] == key) return argv[i + 1];
    return def;
}
inline bool has_flag(int argc, char** argv, const std::string& key) {
    for (int i = 1; i < argc; ++i)
        if (argv[i] == key) return true;
    return false;
}

}  // namespace bench

// === C:\Users\Cutie Magic 500\projects\creative\screen-ocr\include\dks\dks.hpp ===
#pragma once
// dks — deterministic screen segmentation + OCR. Header-only, C++17, no dependencies.
//
//   core/      geometry, image views, UTF-8
//   imgproc/   edge mask, Otsu ink extraction
//   segment/   run-based CCL, containment forest, layout (words / lines / icons / containers)
//   ocr/       glyph features, font atlas, k-NN / VP-tree classifier, word & line recognizer
//   eval/      detection / grouping / hierarchy / CER metrics
//   platform/  (opt-in) Win32 capture + WIC image IO — include explicitly
#include "core/geometry.hpp"
#include "core/image.hpp"
#include "core/utf8.hpp"
#include "eval/metrics.hpp"
#include "imgproc/binarize.hpp"
#include "ocr/ocr.hpp"
#include "segment/ccl.hpp"
#include "segment/hierarchy.hpp"
#include "segment/layout.hpp"

// === C:\Users\Cutie Magic 500\projects\creative\screen-ocr\include\dks\pipeline.hpp ===
#pragma once
// ScreenReader: layout analysis + a character-recognition pass over *every* detected box.
//
// The OCR pass doubles as a text detector: a box is kept as text only when its reading looks like
// text (enough glyphs, mostly letters/digits, good template matches, glyphs sitting on a consistent
// baseline). On Zenodo this lifts Text-line precision 0.49 -> 0.80 at ~2 pts recall, and removes
// ~99.9% of false "text" on photographic wallpapers (see README).
//
// Boxes are read in parallel (one Classifier copy per thread; copies share the template index).
// Output is identical for any thread count: each box is recognised independently and the memo
// cache only affects speed.
#include <algorithm>
#include <string>
#include <thread>
#include <vector>

#include "ocr/ocr.hpp"
#include "segment/layout.hpp"

namespace dks {

struct TextVerifyParams {
    bool enabled = true;
    int min_glyphs = 3;             // 1-2 glyph readings are indistinguishable from pictograms / texture
    float min_alnum = 0.7f;         // share of letters/digits
    float max_mean_dist = 22.f;     // mean template distance of the chosen labels
    float max_fit_residual = 0.5f;  // mean baseline/cap-height placement error (cap heights)
    int max_height = 80;            // taller boxes are not read as a single line
};

struct ReadElement {
    Element element;       // geometry, kind, hierarchy (from analyze_layout)
    std::string text;      // OCR reading of the box (every kind is read)
    float confidence = 0;  // 0..1 text-likeness of the reading
    bool is_text = false;  // passed verification
    ocr::Recognition detail;
};

struct ScreenRead {
    Layout layout;
    std::vector<ReadElement> elements;  // same order as layout.elements; rejected boxes are kept, is_text=false
};

inline float text_confidence(const ocr::Recognition& r, const TextVerifyParams& V) {
    if (r.glyphs.empty()) return 0.f;
    const float shape = std::clamp(1.f - r.mean_dist / V.max_mean_dist, 0.f, 1.f);
    const float fit = r.line_fit ? std::clamp(1.f - r.fit_residual / V.max_fit_residual, 0.f, 1.f) : 0.5f;
    const float count = std::min(1.f, float(r.glyphs.size()) / float(std::max(1, V.min_glyphs)));
    return std::clamp(0.35f * shape + 0.35f * r.alnum_frac + 0.15f * fit + 0.15f * count, 0.f, 1.f);
}

inline bool passes_verify(const ocr::Recognition& r, const TextVerifyParams& V) {
    if (int(r.glyphs.size()) < V.min_glyphs) return false;
    if (r.alnum_frac < V.min_alnum) return false;
    if (r.mean_dist > V.max_mean_dist) return false;
    if (r.line_fit && r.fit_residual > V.max_fit_residual) return false;
    return true;
}

class ScreenReader {
public:
    // threads = 0 -> hardware concurrency.
    ScreenReader(const ocr::Classifier& cls, LayoutParams lp = {}, ocr::RecognizerParams rp = {}, TextVerifyParams vp = {},
                 unsigned threads = 0)
        : rp_(rp), lp_(lp), vp_(vp) {
        const unsigned n = threads ? threads : std::max(1u, std::thread::hardware_concurrency());
        cls_.reserve(n);
        for (unsigned i = 0; i < n; ++i) cls_.push_back(cls);
    }

    LayoutParams& layout_params() noexcept { return lp_; }
    TextVerifyParams& verify_params() noexcept { return vp_; }
    unsigned threads() const noexcept { return unsigned(cls_.size()); }
    void enable_memo(bool on) {
        for (auto& c : cls_) c.enable_memo(on);
    }

    ScreenRead read(const ColorView& frame) const {
        ScreenRead out;
        out.layout = analyze_layout(frame, lp_);
        const Gray8 gray = to_luma(frame);
        out.elements.resize(out.layout.elements.size());
        auto work = [&](size_t t) {
            const ocr::Recognizer rec(cls_[t], rp_);
            // Static striping: box i always lands on thread i % T, so each thread's memo sees the
            // same boxes frame after frame (dynamic scheduling scatters them and misses the cache).
            for (size_t i = t; i < out.elements.size(); i += cls_.size()) {
                ReadElement& re = out.elements[i];
                const Element& e = out.layout.elements[i];
                re.element = e;
                if (e.bbox.h > vp_.max_height) continue;
                const Rect r = e.bbox.inflate(2).clip(gray.width(), gray.height());
                re.detail = rec.recognize(gray.cview().sub(r));
                re.text = re.detail.utf8();
                re.confidence = text_confidence(re.detail, vp_);
                re.is_text = vp_.enabled ? passes_verify(re.detail, vp_) : e.kind == ElementKind::Text;
            }
        };
        if (cls_.size() == 1) {
            work(0);
        } else {
            std::vector<std::thread> pool;
            for (size_t t = 0; t < cls_.size(); ++t) pool.emplace_back(work, t);
            for (auto& th : pool) th.join();
        }
        return out;
    }

    // Aggregate classifier counters over all thread copies.
    uint64_t memo_hits() const {
        uint64_t s = 0;
        for (const auto& c : cls_) s += c.memo_hits;
        return s;
    }

private:
    mutable std::vector<ocr::Classifier> cls_;
    ocr::RecognizerParams rp_;
    LayoutParams lp_;
    TextVerifyParams vp_;
};

}  // namespace dks

// === C:\Users\Cutie Magic 500\projects\creative\screen-ocr\include\dks\core\geometry.hpp ===
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

// === C:\Users\Cutie Magic 500\projects\creative\screen-ocr\include\dks\core\image.hpp ===
#pragma once
// Minimal non-owning image views + owning planar images, and pixel-format -> luma conversion.
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "geometry.hpp"

namespace dks {

template <class T>
struct ImageView {
    T* data = nullptr;
    int32_t width = 0, height = 0;
    ptrdiff_t stride = 0;  // in elements

    T* row(int32_t y) const noexcept { return data + y * stride; }
    T& at(int32_t x, int32_t y) const noexcept { return data[y * stride + x]; }
    bool empty() const noexcept { return data == nullptr || width <= 0 || height <= 0; }

    ImageView sub(const Rect& r) const noexcept {
        const Rect c = r.clip(width, height);
        return ImageView{data + c.y * stride + c.x, c.w, c.h, stride};
    }
    operator ImageView<const T>() const noexcept { return {data, width, height, stride}; }
};

template <class T>
class Image {
public:
    Image() = default;
    Image(int32_t w, int32_t h, T fill = T{}) : w_(w), h_(h), buf_(size_t(w) * size_t(h), fill) {}

    void reset(int32_t w, int32_t h, T fill = T{}) {
        w_ = w;
        h_ = h;
        buf_.assign(size_t(w) * size_t(h), fill);
    }

    int32_t width() const noexcept { return w_; }
    int32_t height() const noexcept { return h_; }
    T* data() noexcept { return buf_.data(); }
    const T* data() const noexcept { return buf_.data(); }
    T* row(int32_t y) noexcept { return buf_.data() + size_t(y) * w_; }
    const T* row(int32_t y) const noexcept { return buf_.data() + size_t(y) * w_; }
    T& at(int32_t x, int32_t y) noexcept { return buf_[size_t(y) * w_ + x]; }
    const T& at(int32_t x, int32_t y) const noexcept { return buf_[size_t(y) * w_ + x]; }

    ImageView<T> view() noexcept { return {buf_.data(), w_, h_, w_}; }
    ImageView<const T> view() const noexcept { return {buf_.data(), w_, h_, w_}; }
    ImageView<const T> cview() const noexcept { return view(); }

private:
    int32_t w_ = 0, h_ = 0;
    std::vector<T> buf_;
};

using Gray8 = Image<uint8_t>;
using GrayView = ImageView<const uint8_t>;

enum class PixelFormat : uint8_t { Gray8, RGB24, BGR24, RGBA32, BGRA32 };

constexpr int bytes_per_pixel(PixelFormat f) noexcept {
    switch (f) {
        case PixelFormat::Gray8: return 1;
        case PixelFormat::RGB24:
        case PixelFormat::BGR24: return 3;
        default: return 4;
    }
}

// Interleaved colour frame (e.g. a screen capture). Non-owning.
struct ColorView {
    const uint8_t* data = nullptr;
    int32_t width = 0, height = 0;
    ptrdiff_t stride_bytes = 0;
    PixelFormat format = PixelFormat::BGRA32;

    const uint8_t* row(int32_t y) const noexcept { return data + y * stride_bytes; }
    // Channel offsets of R, G, B inside one pixel.
    void rgb_offsets(int& r, int& g, int& b) const noexcept {
        switch (format) {
            case PixelFormat::RGB24:
            case PixelFormat::RGBA32: r = 0, g = 1, b = 2; break;
            case PixelFormat::BGR24:
            case PixelFormat::BGRA32: r = 2, g = 1, b = 0; break;
            default: r = g = b = 0; break;
        }
    }
};

// Rec.601 luma in 8.8 fixed point: Y = (77R + 150G + 29B + 128) >> 8. Loop is branch-free and
// auto-vectorises for the fixed-format inner loops.
inline void to_luma(const ColorView& src, ImageView<uint8_t> dst) noexcept {
    assert(dst.width == src.width && dst.height == src.height);
    const int bpp = bytes_per_pixel(src.format);
    if (src.format == PixelFormat::Gray8) {
        for (int32_t y = 0; y < src.height; ++y) {
            const uint8_t* s = src.row(y);
            uint8_t* d = dst.row(y);
            for (int32_t x = 0; x < src.width; ++x) d[x] = s[x];
        }
        return;
    }
    int ro, go, bo;
    src.rgb_offsets(ro, go, bo);
    for (int32_t y = 0; y < src.height; ++y) {
        const uint8_t* s = src.row(y);
        uint8_t* d = dst.row(y);
        for (int32_t x = 0; x < src.width; ++x, s += bpp)
            d[x] = uint8_t((77u * s[ro] + 150u * s[go] + 29u * s[bo] + 128u) >> 8);
    }
}

inline Gray8 to_luma(const ColorView& src) {
    Gray8 out(src.width, src.height);
    to_luma(src, out.view());
    return out;
}

inline Gray8 crop(GrayView src, const Rect& r) {
    const Rect c = r.clip(src.width, src.height);
    Gray8 out(c.w, c.h);
    for (int32_t y = 0; y < c.h; ++y) {
        const uint8_t* s = src.row(c.y + y) + c.x;
        uint8_t* d = out.row(y);
        for (int32_t x = 0; x < c.w; ++x) d[x] = s[x];
    }
    return out;
}

}  // namespace dks

// === C:\Users\Cutie Magic 500\projects\creative\screen-ocr\include\dks\core\utf8.hpp ===
#pragma once
#include <cstdint>
#include <string>
#include <string_view>

namespace dks {

inline std::u32string utf8_decode(std::string_view s) {
    std::u32string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size();) {
        const uint8_t c = uint8_t(s[i]);
        char32_t cp;
        int n;
        if (c < 0x80) cp = c, n = 1;
        else if ((c >> 5) == 0x6) cp = c & 0x1F, n = 2;
        else if ((c >> 4) == 0xE) cp = c & 0x0F, n = 3;
        else if ((c >> 3) == 0x1E) cp = c & 0x07, n = 4;
        else { out.push_back(0xFFFD); ++i; continue; }
        if (i + size_t(n) > s.size()) { out.push_back(0xFFFD); break; }
        for (int k = 1; k < n; ++k) cp = (cp << 6) | (uint8_t(s[i + size_t(k)]) & 0x3F);
        out.push_back(cp);
        i += size_t(n);
    }
    return out;
}

inline void utf8_append(std::string& out, char32_t cp) {
    if (cp < 0x80) out.push_back(char(cp));
    else if (cp < 0x800) out.push_back(char(0xC0 | (cp >> 6))), out.push_back(char(0x80 | (cp & 0x3F)));
    else if (cp < 0x10000)
        out.push_back(char(0xE0 | (cp >> 12))), out.push_back(char(0x80 | ((cp >> 6) & 0x3F))),
            out.push_back(char(0x80 | (cp & 0x3F)));
    else
        out.push_back(char(0xF0 | (cp >> 18))), out.push_back(char(0x80 | ((cp >> 12) & 0x3F))),
            out.push_back(char(0x80 | ((cp >> 6) & 0x3F))), out.push_back(char(0x80 | (cp & 0x3F)));
}

inline std::string utf8_encode(std::u32string_view s) {
    std::string out;
    for (char32_t c : s) utf8_append(out, c);
    return out;
}

}  // namespace dks

// === C:\Users\Cutie Magic 500\projects\creative\screen-ocr\include\dks\eval\metrics.hpp ===
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

// === C:\Users\Cutie Magic 500\projects\creative\screen-ocr\include\dks\imgproc\binarize.hpp ===
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

// === C:\Users\Cutie Magic 500\projects\creative\screen-ocr\include\dks\ocr\atlas.hpp ===
#pragma once
// Glyph template atlas: loads the raw rendered glyphs (tools/build_atlas.py), pushes every bitmap
// through the live feature pipeline, and prunes near-duplicate templates per character.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "glyph_cut.hpp"

namespace dks::ocr {

struct Template {
    GlyphFeature f;
    char32_t ch = 0;
    uint16_t font = 0, px = 0;
    float top_rel = 0, bot_rel = 0;  // mask top / bottom relative to the baseline, in cap heights (+ = down)
};

class Atlas {
public:
    std::vector<std::string> fonts;
    std::vector<Template> templates;
    size_t raw_count = 0;

    bool load(const std::string& path, std::string* err = nullptr) {
        FILE* fp = std::fopen(path.c_str(), "rb");
        if (!fp) return fail(err, "cannot open " + path);
        std::vector<uint8_t> buf;
        uint8_t tmp[1 << 16];
        size_t n;
        while ((n = std::fread(tmp, 1, sizeof tmp, fp)) > 0) buf.insert(buf.end(), tmp, tmp + n);
        std::fclose(fp);

        size_t p = 0;
        auto need = [&](size_t k) { return p + k <= buf.size(); };
        auto u16 = [&] { uint16_t v = uint16_t(buf[p] | (buf[p + 1] << 8)); p += 2; return v; };
        auto u32 = [&] {
            uint32_t v = uint32_t(buf[p]) | (uint32_t(buf[p + 1]) << 8) | (uint32_t(buf[p + 2]) << 16) |
                         (uint32_t(buf[p + 3]) << 24);
            p += 4;
            return v;
        };
        if (!need(8) || std::string(buf.begin(), buf.begin() + 4) != "DKSA") return fail(err, "bad magic");
        p = 4;
        if (u32() != 1) return fail(err, "unsupported version");
        const uint32_t nf = u32();
        for (uint32_t i = 0; i < nf; ++i) {
            const uint16_t len = u16();
            if (!need(len)) return fail(err, "truncated");
            fonts.emplace_back(buf.begin() + ptrdiff_t(p), buf.begin() + ptrdiff_t(p + len));
            p += len;
        }
        const uint32_t ng = u32();
        templates.reserve(ng);
        constexpr int32_t pad = 2;
        for (uint32_t i = 0; i < ng; ++i) {
            if (!need(18)) return fail(err, "truncated");
            Template t;
            t.ch = u32();
            t.font = u16();
            t.px = u16();
            const int16_t top = int16_t(u16());
            const uint16_t w = u16(), h = u16(), cap_h = u16();
            u16();  // x-height (unused for now)
            if (!need(size_t(w) * h)) return fail(err, "truncated");
            // Dark-on-light padded crop so polarity / Otsu behave exactly as on a screen crop.
            Gray8 g(w + 2 * pad, h + 2 * pad, 255);
            for (uint16_t y = 0; y < h; ++y)
                for (uint16_t x = 0; x < w; ++x) g.at(x + pad, y + pad) = uint8_t(255 - buf[p + size_t(y) * w + x]);
            p += size_t(w) * h;

            const CropContext C = analyze_crop(g.cview());
            if (C.ccl.components.empty()) continue;
            std::vector<Source> src;
            for (uint32_t k = 0; k < C.ccl.components.size(); ++k) src.push_back({k, INT32_MIN, INT32_MAX});
            Rect bb;
            t.f = sources_feature(C, src.data(), src.data() + src.size(), &bb);
            const float ch = float(std::max<uint16_t>(1, cap_h));
            t.top_rel = float(top + bb.y - pad) / ch;
            t.bot_rel = float(top + bb.bottom() - pad) / ch;
            templates.push_back(t);
        }
        raw_count = templates.size();
        return true;
    }

    // Greedy per-character pruning: drop a template within `tau` of an already-kept one.
    void prune(float tau, const FeatureWeights& w = {}) {
        std::stable_sort(templates.begin(), templates.end(), [](const Template& a, const Template& b) {
            return a.ch != b.ch ? a.ch < b.ch : a.px > b.px;
        });
        std::vector<Template> kept;
        kept.reserve(templates.size());
        size_t first_of_char = 0;
        for (const Template& t : templates) {
            if (kept.empty() || kept.back().ch != t.ch) first_of_char = kept.size();
            bool dup = false;
            for (size_t j = first_of_char; j < kept.size() && !dup; ++j) dup = distance(t.f, kept[j].f, w) < tau;
            if (!dup) kept.push_back(t);
        }
        templates.swap(kept);
    }

private:
    static bool fail(std::string* err, const std::string& m) {
        if (err) *err = m;
        return false;
    }
};

}  // namespace dks::ocr

// === C:\Users\Cutie Magic 500\projects\creative\screen-ocr\include\dks\ocr\classifier.hpp ===
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

// === C:\Users\Cutie Magic 500\projects\creative\screen-ocr\include\dks\ocr\features.hpp ===
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

// === C:\Users\Cutie Magic 500\projects\creative\screen-ocr\include\dks\ocr\glyph_cut.hpp ===
#pragma once
// Cutting glyph hypotheses out of a text crop.
//
// A crop is binarised once (Otsu + border polarity vote) and labelled once. A glyph hypothesis is a
// set of Sources: (component, [xlo, xhi) column window). Column windows let one touching-glyph
// component be split at a cut column without re-labelling anything.
#include <algorithm>
#include <cstdint>
#include <vector>

#include "../imgproc/binarize.hpp"
#include "../segment/ccl.hpp"
#include "features.hpp"

namespace dks::ocr {

struct Source {
    uint32_t comp;
    int32_t xlo, xhi;
};

struct CropContext {
    InkResult ink;
    LabelResult ccl;
    int32_t width = 0, height = 0;
};

inline CropContext analyze_crop(GrayView gray, const InkParams& ip = {}) {
    CropContext c;
    c.width = gray.width, c.height = gray.height;
    c.ink = extract_ink(gray, ip);
    c.ccl = label_components(c.ink.mask.cview(), Connectivity::Eight);
    return c;
}

// Exact bbox of the pixels selected by `srcs`.
inline Rect sources_bbox(const CropContext& C, const Source* b, const Source* e) {
    Rect r;
    for (const Source* s = b; s != e; ++s) {
        const Component& k = C.ccl.components[s->comp];
        if (s->xlo <= k.bbox.x && s->xhi >= k.bbox.right()) { r = r.unite(k.bbox); continue; }
        for (uint32_t i = k.run_begin; i < k.run_end; ++i) {
            const Run& run = C.ccl.runs[i];
            const int32_t x0 = std::max(run.x0, s->xlo), x1 = std::min(run.x1, s->xhi);
            if (x1 > x0) r = r.unite(Rect{x0, run.y, x1 - x0, 1});
        }
    }
    return r;
}

// Feature of a glyph hypothesis. Soft ink is taken from the glyph's own pixels plus the 1px
// antialias fringe around them that no other glyph claims.
inline GlyphFeature sources_feature(const CropContext& C, const Source* b, const Source* e, Rect* bbox_out = nullptr) {
    const Rect bb = sources_bbox(C, b, e);
    if (bbox_out) *bbox_out = bb;
    if (bb.empty()) return GlyphFeature{};
    Gray8 m(bb.w, bb.h, 0), ink(bb.w, bb.h, 0);
    for (const Source* s = b; s != e; ++s) {
        const Component& k = C.ccl.components[s->comp];
        for (uint32_t i = k.run_begin; i < k.run_end; ++i) {
            const Run& run = C.ccl.runs[i];
            const int32_t x0 = std::max(run.x0, s->xlo), x1 = std::min(run.x1, s->xhi);
            uint8_t* row = m.row(run.y - bb.y);
            for (int32_t x = x0; x < x1; ++x) row[x - bb.x] = 1;
        }
    }
    const Gray8& gm = C.ink.mask;
    const Gray8& gi = C.ink.ink;
    for (int32_t y = 0; y < bb.h; ++y) {
        for (int32_t x = 0; x < bb.w; ++x) {
            const int32_t X = bb.x + x, Y = bb.y + y;
            if (m.at(x, y)) { ink.at(x, y) = gi.at(X, Y); continue; }
            if (gm.at(X, Y)) continue;  // someone else's ink
            bool fringe = false;
            for (int dy = -1; dy <= 1 && !fringe; ++dy)
                for (int dx = -1; dx <= 1 && !fringe; ++dx) {
                    const int32_t xx = x + dx, yy = y + dy;
                    fringe = xx >= 0 && yy >= 0 && xx < bb.w && yy < bb.h && m.at(xx, yy);
                }
            if (fringe) ink.at(x, y) = gi.at(X, Y);
        }
    }
    return make_feature(ink.cview(), m.cview());
}

}  // namespace dks::ocr

// === C:\Users\Cutie Magic 500\projects\creative\screen-ocr\include\dks\ocr\ocr.hpp ===
#pragma once
// OCR umbrella: glyph features, atlas, classifier, recogniser.
#include "atlas.hpp"
#include "classifier.hpp"
#include "features.hpp"
#include "glyph_cut.hpp"
#include "recognizer.hpp"

// === C:\Users\Cutie Magic 500\projects\creative\screen-ocr\include\dks\ocr\recognizer.hpp ===
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

// === C:\Users\Cutie Magic 500\projects\creative\screen-ocr\include\dks\ocr\simd_metric.hpp ===
#pragma once
// SIMD kernels for the 16x16 glyph metric. Exact integer results, identical on every path
// (AVX2 / SSE2 / scalar), so recognition stays deterministic regardless of the build target.
#include <cstdint>
#include <cstdlib>

#if defined(__AVX2__)
#include <immintrin.h>
#define DKS_SIMD_AVX2 1
#elif defined(__SSE2__) || defined(_M_X64)
#include <emmintrin.h>
#define DKS_SIMD_SSE2 1
#endif

namespace dks::ocr {

// Sum of absolute differences over [begin, end) bytes; begin/end multiples of 32.
inline uint32_t sad_range(const uint8_t* __restrict a, const uint8_t* __restrict b, int begin, int end) noexcept {
#if defined(DKS_SIMD_AVX2)
    __m256i acc = _mm256_setzero_si256();
    for (int i = begin; i < end; i += 32) {
        const __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
        const __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
        acc = _mm256_add_epi64(acc, _mm256_sad_epu8(va, vb));
    }
    const __m128i s = _mm_add_epi64(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
    return uint32_t(_mm_cvtsi128_si32(s)) + uint32_t(_mm_cvtsi128_si32(_mm_srli_si128(s, 8)));
#elif defined(DKS_SIMD_SSE2)
    __m128i acc = _mm_setzero_si128();
    for (int i = begin; i < end; i += 16) {
        const __m128i va = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + i));
        const __m128i vb = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b + i));
        acc = _mm_add_epi64(acc, _mm_sad_epu8(va, vb));
    }
    return uint32_t(_mm_cvtsi128_si32(acc)) + uint32_t(_mm_cvtsi128_si32(_mm_srli_si128(acc, 8)));
#else
    uint32_t s = 0;
    for (int i = begin; i < end; ++i) s += uint32_t(std::abs(int(a[i]) - int(b[i])));
    return s;
#endif
}

inline uint32_t sad256_simd(const uint8_t* __restrict a, const uint8_t* __restrict b) noexcept {
    return sad_range(a, b, 0, 256);
}

// Early-abandon SAD: stops after the first 64-byte quarter whose running sum exceeds `limit`
// (returns that partial sum, which is already > limit, so the caller's rejection is exact).
inline uint32_t sad256_abandon(const uint8_t* __restrict a, const uint8_t* __restrict b, uint32_t limit) noexcept {
    uint32_t s = 0;
    for (int q = 0; q < 256; q += 64) {
        s += sad_range(a, b, q, q + 64);
        if (s > limit) return s;
    }
    return s;
}

}  // namespace dks::ocr

// === C:\Users\Cutie Magic 500\projects\creative\screen-ocr\include\dks\platform\win32.hpp ===
#pragma once
// Optional Windows platform layer (not included by dks.hpp):
//   capture_screen()  GDI BitBlt of the virtual desktop -> BGRA frame
//   load_image()      WIC decode of PNG/JPEG/GIF/BMP/WebP/TIFF -> BGRA frame
//   save_png()        WIC encode BGRA -> PNG (for debug overlays)
// Link: -lgdi32 -lole32 -lwindowscodecs -luuid
#ifndef _WIN32
#error "dks/platform/win32.hpp is Windows-only"
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <wincodec.h>

#include <cstdint>
#include <string>
#include <vector>

#include "../core/image.hpp"

namespace dks::win32 {

struct Frame {
    std::vector<uint8_t> pixels;  // BGRA, tightly packed
    int32_t width = 0, height = 0;
    ColorView view() const noexcept {
        return ColorView{pixels.data(), width, height, ptrdiff_t(width) * 4, PixelFormat::BGRA32};
    }
    bool empty() const noexcept { return pixels.empty(); }
};

inline std::wstring widen(const std::string& s) {
    if (s.empty()) return {};
    const int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), int(s.size()), nullptr, 0);
    std::wstring w(size_t(n), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.data(), int(s.size()), w.data(), n);
    return w;
}

// Captures the full virtual screen (all monitors). Per-monitor DPI awareness should be enabled by
// the caller (SetProcessDpiAwarenessContext) to get physical pixels.
inline Frame capture_screen() {
    Frame f;
    const int x = GetSystemMetrics(SM_XVIRTUALSCREEN), y = GetSystemMetrics(SM_YVIRTUALSCREEN);
    const int w = GetSystemMetrics(SM_CXVIRTUALSCREEN), h = GetSystemMetrics(SM_CYVIRTUALSCREEN);
    HDC screen = GetDC(nullptr);
    HDC mem = CreateCompatibleDC(screen);
    BITMAPINFO bi{};
    bi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    bi.bmiHeader.biWidth = w;
    bi.bmiHeader.biHeight = -h;  // top-down
    bi.bmiHeader.biPlanes = 1;
    bi.bmiHeader.biBitCount = 32;
    bi.bmiHeader.biCompression = BI_RGB;
    void* bits = nullptr;
    HBITMAP dib = CreateDIBSection(screen, &bi, DIB_RGB_COLORS, &bits, nullptr, 0);
    if (dib && bits) {
        HGDIOBJ old = SelectObject(mem, dib);
        BitBlt(mem, 0, 0, w, h, screen, x, y, SRCCOPY | CAPTUREBLT);
        GdiFlush();
        f.width = w, f.height = h;
        f.pixels.assign(static_cast<uint8_t*>(bits), static_cast<uint8_t*>(bits) + size_t(w) * h * 4);
        SelectObject(mem, old);
    }
    if (dib) DeleteObject(dib);
    DeleteDC(mem);
    ReleaseDC(nullptr, screen);
    return f;
}

namespace detail {
struct ComInit {
    ComInit() { CoInitializeEx(nullptr, COINIT_MULTITHREADED); }
    ~ComInit() { CoUninitialize(); }
};
template <class T>
struct Com {
    T* p = nullptr;
    ~Com() { if (p) p->Release(); }
    T** operator&() { return &p; }
    T* operator->() { return p; }
};
inline IWICImagingFactory* factory() {
    static ComInit init;
    static IWICImagingFactory* f = [] {
        IWICImagingFactory* fac = nullptr;
        CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&fac));
        return fac;
    }();
    return f;
}
}  // namespace detail

inline Frame load_image(const std::string& utf8_path) {
    Frame f;
    IWICImagingFactory* fac = detail::factory();
    if (!fac) return f;
    detail::Com<IWICBitmapDecoder> dec;
    if (FAILED(fac->CreateDecoderFromFilename(widen(utf8_path).c_str(), nullptr, GENERIC_READ,
                                              WICDecodeMetadataCacheOnDemand, &dec)))
        return f;
    detail::Com<IWICBitmapFrameDecode> frame;
    if (FAILED(dec->GetFrame(0, &frame))) return f;
    detail::Com<IWICFormatConverter> conv;
    if (FAILED(fac->CreateFormatConverter(&conv))) return f;
    if (FAILED(conv->Initialize(frame.p, GUID_WICPixelFormat32bppBGRA, WICBitmapDitherTypeNone, nullptr, 0.0,
                                WICBitmapPaletteTypeCustom)))
        return f;
    UINT w = 0, h = 0;
    conv->GetSize(&w, &h);
    f.width = int32_t(w), f.height = int32_t(h);
    f.pixels.resize(size_t(w) * h * 4);
    if (FAILED(conv->CopyPixels(nullptr, w * 4, UINT(f.pixels.size()), f.pixels.data()))) f = Frame{};
    return f;
}

inline bool save_png(const std::string& utf8_path, const Frame& img) {
    IWICImagingFactory* fac = detail::factory();
    if (!fac || img.empty()) return false;
    detail::Com<IWICStream> stream;
    if (FAILED(fac->CreateStream(&stream))) return false;
    if (FAILED(stream->InitializeFromFilename(widen(utf8_path).c_str(), GENERIC_WRITE))) return false;
    detail::Com<IWICBitmapEncoder> enc;
    if (FAILED(fac->CreateEncoder(GUID_ContainerFormatPng, nullptr, &enc))) return false;
    if (FAILED(enc->Initialize(stream.p, WICBitmapEncoderNoCache))) return false;
    detail::Com<IWICBitmapFrameEncode> fr;
    if (FAILED(enc->CreateNewFrame(&fr, nullptr))) return false;
    fr->Initialize(nullptr);
    fr->SetSize(UINT(img.width), UINT(img.height));
    WICPixelFormatGUID fmt = GUID_WICPixelFormat32bppBGRA;
    fr->SetPixelFormat(&fmt);
    fr->WritePixels(UINT(img.height), UINT(img.width * 4), UINT(img.pixels.size()),
                    const_cast<BYTE*>(img.pixels.data()));
    fr->Commit();
    return SUCCEEDED(enc->Commit());
}

}  // namespace dks::win32

// === C:\Users\Cutie Magic 500\projects\creative\screen-ocr\include\dks\segment\ccl.hpp ===
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

// === C:\Users\Cutie Magic 500\projects\creative\screen-ocr\include\dks\segment\hierarchy.hpp ===
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

// === C:\Users\Cutie Magic 500\projects\creative\screen-ocr\include\dks\segment\layout.hpp ===
#pragma once
// Whole-screen layout analysis:
//   colour frame -> edge mask -> CCL atoms -> {glyph, container, icon, image} -> words -> lines
//   -> containment forest over all elements.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

#include "../core/image.hpp"
#include "../imgproc/binarize.hpp"
#include "ccl.hpp"
#include "hierarchy.hpp"

namespace dks {

struct LayoutParams {
    int edge_threshold = 28;       // per-channel step that counts as structure
    int glyph_min_h = 5;           // px
    int glyph_max_h = 72;          // px, bigger is display text / image
    float glyph_max_aspect = 12.f; // w/h; touching-glyph clusters are wide
    int glyph_max_tall = 12;       // h/w above this is a rule line, not a glyph (a 1px 'l' is ~8-10)
    float word_gap = 0.55f;        // x-gap / line height to join atoms into a word
    float line_gap = 1.25f;        // x-gap / line height to join words into a line
    float block_leading = 0.9f;
    // Merge vetoes (see analyze_layout):
    bool gutter_veto = true;       // no line merge across a blank column band (whitespace river)
    float gutter_min_gap = 0.6f;   // ... only for gaps wider than this * height (word spaces are exempt)
    float gutter_reach = 2.0f;     // band extends this * height above and below the pair
    bool luma_veto = true;         // no merge between atoms/words whose ink luma differs > luma_delta
    int luma_delta = 35;
    int weak_edge_threshold = 10;  // hysteresis low threshold for flat-region containers (0 = off)    // vertical gap / line height to stack lines into a paragraph block
    float min_v_overlap = 0.55f;   // vertical overlap / min height
    float max_h_ratio = 2.6f;      // height ratio for two atoms in the same line
    int container_min_w = 24, container_min_h = 14;
    float container_max_fill = 0.22f;
    float container_border_frac = 0.70f;  // share of ink within `border_band` px of the bbox edge
    int border_band = 3;
    int icon_min = 9, icon_max = 96;
    int icon_colors = 8;       // distinct 12-bit colours marking a line-end atom as a pictogram
    float icon_h_ratio = 1.2f;
    // Icon saliency: real UI icons are high-contrast marks on a calm surround; texture fragments of
    // photos / gradient wallpapers are low-contrast and surrounded by more texture.
    int icon_min_contrast = 40;       // p98 - p2 luma inside the box
    float icon_max_ring_std = 12.f;   // max per-channel std-dev of a 3px ring around the box // ... when it is also this much taller than the rest of the line
    int icon_merge_gap = 3;    // px: pictogram fragments closer than this form one icon
    int min_area = 3;
    // Flat-region containers: 4-connected non-edge regions (panels, buttons, fields) that fill most
    // of their bounding box, with or without a drawn border.
    bool flat_containers = true;
    float flat_min_fill = 0.55f;
    // Images: cells with many distinct quantised colours (photos / illustrations; text and flat UI
    // use a small palette even with antialiasing).
    bool detect_images = true;
    int image_cell = 16;
    int image_colors = 40;     // distinct 12-bit colours per cell
    int image_min_cells = 6;
    float image_min_edge_density = 0.05f;  // busy cell also needs real structure (smooth 3D shading has
                                           // many colours but almost no edges)
    int image_suppress_min = 100;          // only images at least this big hide the atoms inside them
                                           // (small busy regions are icon+label pairs, avatars, ...)
    float image_min_fill = 0.35f;          // busy cells / bbox cells of an image region
    int image_min_size = 0;                // >0: smaller busy regions are left to the icon detector
    float flat_max_std = 6.f;              // flat-region container: max luma std-dev of its own pixels
};

enum class ElementKind : uint8_t { Text, Icon, Image, Container };

inline const char* kind_name(ElementKind k) noexcept {
    switch (k) {
        case ElementKind::Text: return "Text";
        case ElementKind::Icon: return "Icon";
        case ElementKind::Image: return "Image";
        default: return "Container";
    }
}

struct Element {
    Rect bbox;
    ElementKind kind = ElementKind::Text;
    int32_t parent = -1, depth = 0;
    std::vector<uint32_t> atoms;  // CCL component ids (text: glyph atoms sorted by x)
};

struct Layout {
    LabelResult ccl;
    std::vector<Element> elements;  // lines (Text), icons, images, containers
    std::vector<Element> words;     // finer Text granularity
    std::vector<Element> blocks;    // coarser Text granularity: paragraphs (stacked, aligned lines)
    Forest forest;                  // over `elements`
};

namespace detail {

inline float border_fraction(const LabelResult& L, const Component& c, int band) {
    const Rect& b = c.bbox;
    int64_t on_border = 0;
    for (uint32_t i = c.run_begin; i < c.run_end; ++i) {
        const Run& r = L.runs[i];
        if (r.y < b.y + band || r.y >= b.bottom() - band) { on_border += r.x1 - r.x0; continue; }
        on_border += std::max(0, std::min(r.x1, b.x + band) - r.x0);
        on_border += std::max(0, r.x1 - std::max(r.x0, b.right() - band));
    }
    return c.area ? float(on_border) / float(c.area) : 0.f;
}

// Union-find grouping of boxes sorted by x: join a->b when on the same line and gap small enough.
template <class Veto>
inline std::vector<std::vector<uint32_t>> group_line_atoms(const std::vector<Rect>& boxes,
                                                           const std::vector<uint32_t>& ids, float gap_k,
                                                           const LayoutParams& P, Veto veto) {
    std::vector<uint32_t> ord(ids);
    std::sort(ord.begin(), ord.end(), [&](uint32_t a, uint32_t b) {
        return boxes[a].x != boxes[b].x ? boxes[a].x < boxes[b].x : boxes[a].y < boxes[b].y;
    });
    std::vector<uint32_t> par(ord.size());
    std::iota(par.begin(), par.end(), 0u);
    auto find = [&](uint32_t i) { while (par[i] != i) i = par[i] = par[par[i]]; return i; };

    for (size_t i = 0; i < ord.size(); ++i) {
        const Rect& a = boxes[ord[i]];
        const int32_t reach = a.right() + int32_t(gap_k * float(a.h) + 0.5f);
        for (size_t j = i + 1; j < ord.size() && boxes[ord[j]].x <= reach; ++j) {
            const Rect& b = boxes[ord[j]];
            const int32_t hmin = std::min(a.h, b.h), hmax = std::max(a.h, b.h);
            if (float(hmax) > P.max_h_ratio * float(hmin)) continue;
            if (float(a.y_overlap(b)) < P.min_v_overlap * float(hmin)) continue;
            const int32_t gap = b.x - a.right();
            if (float(gap) > gap_k * float(hmax)) continue;
            if (veto(ord[i], ord[j])) continue;
            const uint32_t ra = find(uint32_t(i)), rb = find(uint32_t(j));
            if (ra != rb) par[std::max(ra, rb)] = std::min(ra, rb);
        }
    }
    std::vector<std::vector<uint32_t>> groups;
    std::vector<int32_t> slot(ord.size(), -1);
    for (size_t i = 0; i < ord.size(); ++i) {
        const uint32_t r = find(uint32_t(i));
        if (slot[r] < 0) slot[r] = int32_t(groups.size()), groups.emplace_back();
        groups[size_t(slot[r])].push_back(ord[i]);
    }
    return groups;
}


inline int distinct_colors(const ColorView& f, const Rect& r, int cap) {
    uint64_t seen[64] = {};
    int n = 0;
    const int bpp = bytes_per_pixel(f.format);
    int ro, go, bo;
    f.rgb_offsets(ro, go, bo);
    for (int32_t y = r.y; y < r.bottom() && n < cap; ++y) {
        const uint8_t* row = f.row(y);
        for (int32_t x = r.x; x < r.right(); ++x) {
            const uint8_t* px = row + x * bpp;
            const uint32_t key = (uint32_t(px[ro] >> 4) << 8) | (uint32_t(px[go] >> 4) << 4) | uint32_t(px[bo] >> 4);
            const uint64_t bit = 1ull << (key & 63);
            if (!(seen[key >> 6] & bit)) seen[key >> 6] |= bit, ++n;
        }
    }
    return n;
}

// Ink luma of a component: mean luma of its pixels that deviate most from the bbox-border background.
inline int ink_luma(const LabelResult& L, const Component& c, const Gray8& luma) {
    const Rect b = c.bbox;
    std::vector<int> ring;
    for (int32_t x = b.x; x < b.right(); ++x) ring.push_back(luma.at(x, b.y)), ring.push_back(luma.at(x, b.bottom() - 1));
    for (int32_t y = b.y; y < b.bottom(); ++y) ring.push_back(luma.at(b.x, y)), ring.push_back(luma.at(b.right() - 1, y));
    std::nth_element(ring.begin(), ring.begin() + ptrdiff_t(ring.size() / 2), ring.end());
    const int bg = ring[ring.size() / 2];
    int best = 0;
    std::vector<int> v;
    for (uint32_t i = c.run_begin; i < c.run_end; ++i) {
        const Run& r = L.runs[i];
        for (int32_t x = r.x0; x < r.x1; ++x) {
            const int d = luma.at(x, r.y) - bg;
            v.push_back(d);
            best = std::max(best, std::abs(d));
        }
    }
    long sum = 0, n = 0;
    for (int d : v)
        if (2 * std::abs(d) >= best) sum += d, ++n;
    return std::clamp(bg + int(n ? sum / n : 0), 0, 255);
}

// A blank vertical band of >= 3 columns between a and b, extending `reach` heights above and below.
inline bool vertical_gutter(const Gray8& edges, const Rect& a, const Rect& b, float reach) {
    const Rect& l = a.x <= b.x ? a : b;
    const Rect& r = a.x <= b.x ? b : a;
    const int32_t x0 = l.right(), x1 = r.x;
    if (x1 - x0 < 3) return false;
    const int32_t h = std::max(a.h, b.h);
    const int32_t y0 = std::max(0, std::min(a.y, b.y) - int32_t(reach * float(h)));
    const int32_t y1 = std::min(edges.height(), std::max(a.bottom(), b.bottom()) + int32_t(reach * float(h)));
    int run = 0;
    for (int32_t x = x0; x < x1; ++x) {
        bool blank = true;
        for (int32_t y = y0; y < y1 && blank; ++y) blank = edges.at(x, y) == 0;
        run = blank ? run + 1 : 0;
        if (run >= 3) return true;
    }
    return false;
}

// Hysteresis edge map: weak edges (> lo) kept only when 8-connected to a strong edge.
inline Gray8 hysteresis_edges(const ColorView& f, const Gray8& strong, int lo) {
    Gray8 weak(f.width, f.height);
    edge_mask(f, weak.view(), lo);
    const LabelResult L = label_components(weak.cview(), Connectivity::Eight);
    Gray8 out(f.width, f.height, 0);
    for (uint32_t c = 0; c < L.components.size(); ++c) {
        const Component& k = L.components[c];
        bool has_strong = false;
        for (uint32_t i = k.run_begin; i < k.run_end && !has_strong; ++i) {
            const Run& r = L.runs[i];
            for (int32_t x = r.x0; x < r.x1 && !has_strong; ++x) has_strong = strong.at(x, r.y) != 0;
        }
        if (has_strong) L.rasterize(c, out.view());
    }
    return out;
}

inline bool icon_is_salient(const ColorView& f, const Gray8& luma, const Rect& b, const LayoutParams& P) {
    int hist[256] = {};
    int n = 0;
    for (int32_t y = b.y; y < b.bottom(); ++y)
        for (int32_t x = b.x; x < b.right(); ++x) ++hist[luma.at(x, y)], ++n;
    auto pct = [&](int q) {
        int acc = 0;
        for (int v = 0; v < 256; ++v)
            if ((acc += hist[v]) * 100 >= q * n) return v;
        return 255;
    };
    if (pct(98) - pct(2) < P.icon_min_contrast) return false;
    const Rect o = b.inflate(3).clip(f.width, f.height);
    const int bpp = bytes_per_pixel(f.format);
    int ro, go, bo;
    f.rgb_offsets(ro, go, bo);
    double s[3] = {}, s2[3] = {};
    long m = 0;
    for (int32_t y = o.y; y < o.bottom(); ++y) {
        const uint8_t* row = f.row(y);
        for (int32_t x = o.x; x < o.right(); ++x) {
            if (b.contains_point(x, y)) continue;
            const uint8_t* px = row + x * bpp;
            const int c[3] = {px[ro], px[go], px[bo]};
            for (int k = 0; k < 3; ++k) s[k] += c[k], s2[k] += double(c[k]) * c[k];
            ++m;
        }
    }
    if (m == 0) return true;
    double worst = 0;
    for (int k = 0; k < 3; ++k) worst = std::max(worst, s2[k] / double(m) - (s[k] / double(m)) * (s[k] / double(m)));
    return std::sqrt(std::max(0.0, worst)) <= double(P.icon_max_ring_std);
}

// Busy-colour cells -> connected regions -> rectangles tightened on edge density.
inline std::vector<Rect> detect_images(const ColorView& f, const Gray8& edges, const LayoutParams& P) {
    const int cs = P.image_cell;
    const int32_t gw = (f.width + cs - 1) / cs, gh = (f.height + cs - 1) / cs;
    Gray8 busy(gw, gh, 0);
    const int bpp = bytes_per_pixel(f.format);
    int ro, go, bo;
    f.rgb_offsets(ro, go, bo);
    uint64_t seen[64];
    for (int32_t cy = 0; cy < gh; ++cy) {
        for (int32_t cx = 0; cx < gw; ++cx) {
            for (auto& w : seen) w = 0;
            int distinct = 0;
            const int32_t y1 = std::min(f.height, (cy + 1) * cs), x1 = std::min(f.width, (cx + 1) * cs);
            for (int32_t y = cy * cs; y < y1 && distinct < P.image_colors; ++y) {
                const uint8_t* row = f.row(y);
                for (int32_t x = cx * cs; x < x1; ++x) {
                    const uint8_t* px = row + x * bpp;
                    const uint32_t key = (uint32_t(px[ro] >> 4) << 8) | (uint32_t(px[go] >> 4) << 4) | uint32_t(px[bo] >> 4);
                    const uint64_t bit = 1ull << (key & 63);
                    if (!(seen[key >> 6] & bit)) seen[key >> 6] |= bit, ++distinct;
                }
            }
            int edges_in = 0;
            for (int32_t y = cy * cs; y < y1; ++y)
                for (int32_t x = cx * cs; x < x1; ++x) edges_in += edges.at(x, y);
            const int area = (y1 - cy * cs) * (x1 - cx * cs);
            busy.at(cx, cy) = distinct >= P.image_colors && float(edges_in) >= P.image_min_edge_density * float(area);
        }
    }
    std::vector<Rect> out;
    const LabelResult L = label_components(busy.cview(), Connectivity::Eight);
    for (const Component& c : L.components) {
        if (c.area < P.image_min_cells || c.bbox.w < 2 || c.bbox.h < 2) continue;
        // Pictures fill their rectangle with busy cells; an antialiased curve (many blend colours)
        // leaves a thin diagonal chain of busy cells in a mostly empty bbox.
        if (float(c.area) < P.image_min_fill * float(c.bbox.area())) continue;
        Rect r = Rect{c.bbox.x * cs, c.bbox.y * cs, c.bbox.w * cs, c.bbox.h * cs}.inflate(cs / 2).clip(f.width, f.height);
        // Tighten each side while its boundary line is nearly edge-free (outside the picture).
        auto col_density = [&](int32_t x) {
            int n = 0;
            for (int32_t y = r.y; y < r.bottom(); ++y) n += edges.at(x, y);
            return float(n) / float(std::max(1, r.h));
        };
        auto row_density = [&](int32_t y) {
            int n = 0;
            for (int32_t x = r.x; x < r.right(); ++x) n += edges.at(x, y);
            return float(n) / float(std::max(1, r.w));
        };
        while (r.w > cs && col_density(r.x) < 0.08f) ++r.x, --r.w;
        while (r.w > cs && col_density(r.right() - 1) < 0.08f) --r.w;
        while (r.h > cs && row_density(r.y) < 0.08f) ++r.y, --r.h;
        while (r.h > cs && row_density(r.bottom() - 1) < 0.08f) --r.h;
        if (std::max(r.w, r.h) < P.image_min_size) continue;
        out.push_back(r);
    }
    return out;
}

// Large flat (edge-free, 4-connected) regions that fill most of their bbox: panels / buttons / fields.
inline std::vector<Rect> detect_flat_regions(const Gray8& edges, const Gray8& luma, const LayoutParams& P) {
    Gray8 inv(edges.width(), edges.height());
    for (int32_t y = 0; y < edges.height(); ++y) {
        const uint8_t* s = edges.row(y);
        uint8_t* d = inv.row(y);
        for (int32_t x = 0; x < edges.width(); ++x) d[x] = s[x] ? 0 : 1;
    }
    std::vector<Rect> out;
    const LabelResult L = label_components(inv.cview(), Connectivity::Four);
    for (const Component& c : L.components) {
        const Rect& b = c.bbox;
        if (b.w < P.container_min_w || b.h < P.container_min_h) continue;
        if (float(c.area) < P.flat_min_fill * float(b.area())) continue;
        // A UI panel is one colour; smooth gradients / shaded 3D art are edge-free but not uniform.
        double sum = 0, sum2 = 0;
        for (uint32_t i = c.run_begin; i < c.run_end; ++i) {
            const Run& r = L.runs[i];
            const uint8_t* row = luma.row(r.y);
            for (int32_t x = r.x0; x < r.x1; ++x) sum += row[x], sum2 += double(row[x]) * row[x];
        }
        const double n = double(c.area), var = sum2 / n - (sum / n) * (sum / n);
        if (std::sqrt(std::max(0.0, var)) > double(P.flat_max_std)) continue;
        out.push_back(b);
    }
    return out;
}

}  // namespace detail

inline Layout analyze_layout(const ColorView& frame, const LayoutParams& P = {}) {
    Layout out;
    Gray8 mask(frame.width, frame.height);
    edge_mask(frame, mask.view(), P.edge_threshold);
    out.ccl = label_components(mask.cview(), Connectivity::Eight);
    const auto& comps = out.ccl.components;

    const Gray8 luma = to_luma(frame);
    std::vector<Rect> images;
    if (P.detect_images) images = detail::detect_images(frame, mask, P);
    auto inside_image = [&](const Rect& b) {
        for (const Rect& im : images)
            if (std::max(im.w, im.h) >= P.image_suppress_min && im.contains_point(b.x + b.w / 2, b.y + b.h / 2) &&
                im.area() > 2 * b.area())
                return true;
        return false;
    };
    for (const Rect& im : images) {
        Element e;
        e.bbox = im;
        e.kind = ElementKind::Image;
        out.elements.push_back(std::move(e));
    }
    auto add_container = [&](const Rect& b, int32_t atom) {
        for (const Element& o : out.elements)
            if (o.kind == ElementKind::Container && o.bbox.iou(b) > 0.9f) return;
        Element e;
        e.bbox = b;
        e.kind = ElementKind::Container;
        if (atom >= 0) e.atoms = {uint32_t(atom)};
        out.elements.push_back(std::move(e));
    };

    std::vector<Rect> boxes(comps.size());
    std::vector<uint32_t> glyphs, small, pictos;
    for (uint32_t i = 0; i < comps.size(); ++i) {
        const Component& c = comps[i];
        const Rect& b = c.bbox;
        boxes[i] = b;
        if (c.area < P.min_area) continue;

        const bool big_enough = b.w >= P.container_min_w && b.h >= P.container_min_h;
        if (big_enough && c.fill() <= P.container_max_fill &&
            detail::border_fraction(out.ccl, c, P.border_band) >= P.container_border_frac) {
            add_container(b, int32_t(i));
            continue;
        }
        if (inside_image(b)) continue;
        if (b.h >= P.glyph_min_h && b.h <= P.glyph_max_h && float(b.w) <= P.glyph_max_aspect * float(b.h) &&
            b.h <= P.glyph_max_tall * std::max(1, b.w)) {
            glyphs.push_back(i);
            continue;
        }
        if (b.h < P.glyph_min_h && b.w <= 3 * P.glyph_min_h) small.push_back(i);  // . , ' - etc.
    }
    if (P.flat_containers)
        for (const Rect& b : detail::detect_flat_regions(
                 P.weak_edge_threshold > 0 ? detail::hysteresis_edges(frame, mask, P.weak_edge_threshold) : mask, luma, P))
            if (!inside_image(b)) add_container(b, -1);

    // Words: tight gaps. Lines: words joined with wider gaps.
    std::vector<int16_t> atom_ink(comps.size(), -1);
    auto ink_of = [&](uint32_t a) {
        if (atom_ink[a] < 0) atom_ink[a] = int16_t(detail::ink_luma(out.ccl, comps[a], luma));
        return int(atom_ink[a]);
    };
    auto atom_veto = [&](uint32_t a, uint32_t b) {
        return P.luma_veto && std::abs(ink_of(a) - ink_of(b)) > P.luma_delta;
    };
    auto words = detail::group_line_atoms(boxes, glyphs, P.word_gap, P, atom_veto);

    // Absorb small punctuation atoms that sit inside a word's vertical band, just right of it.
    std::vector<Rect> wbox(words.size());
    for (size_t w = 0; w < words.size(); ++w)
        for (uint32_t a : words[w]) wbox[w] = wbox[w].unite(boxes[a]);
    for (uint32_t s : small) {
        const Rect& b = boxes[s];
        int32_t best = -1, best_gap = INT32_MAX;
        for (size_t w = 0; w < words.size(); ++w) {
            const Rect& W = wbox[w];
            if (b.y < W.y - 1 || b.bottom() > W.bottom() + W.h / 3) continue;
            const int32_t gap = b.x >= W.right() ? b.x - W.right() : (b.right() <= W.x ? W.x - b.right() : 0);
            if (gap <= int32_t(P.word_gap * float(W.h)) && gap < best_gap) best = int32_t(w), best_gap = gap;
        }
        if (best >= 0) {
            words[size_t(best)].push_back(s);
            wbox[size_t(best)] = wbox[size_t(best)].unite(b);
        }
    }

    // A word group whose first/last atom is taller than the rest *and* multi-coloured starts or ends
    // with a pictogram (e.g. folder icon + label): split it off.
    for (size_t w = 0; w < words.size(); ++w) {
        auto& g = words[w];
        std::sort(g.begin(), g.end(), [&](uint32_t a, uint32_t b) { return boxes[a].x < boxes[b].x; });
        for (int side = 0; side < 2 && g.size() >= 2; ++side) {
            const uint32_t a = side == 0 ? g.front() : g.back();
            std::vector<int32_t> hs;
            for (uint32_t o : g) if (o != a) hs.push_back(boxes[o].h);
            std::nth_element(hs.begin(), hs.begin() + ptrdiff_t(hs.size() / 2), hs.end());
            const Rect& b = boxes[a];
            const float ar = float(b.w) / float(b.h);
            if (float(b.h) >= P.icon_h_ratio * float(hs[hs.size() / 2]) && b.h >= P.icon_min && ar > 0.6f && ar < 1.7f &&
                detail::distinct_colors(frame, b, P.icon_colors) >= P.icon_colors) {
                pictos.push_back(a);
                if (side == 0) g.erase(g.begin()); else g.pop_back();
            }
        }
        wbox[w] = Rect{};
        for (uint32_t a : g) wbox[w] = wbox[w].unite(boxes[a]);
    }

    // Classify word groups; isolated single atoms that look like pictograms become icon fragments.
    std::vector<Rect> text_word_boxes;
    std::vector<uint32_t> text_word_ids;
    for (size_t w = 0; w < words.size(); ++w) {
        auto& g = words[w];
        if (g.empty()) continue;
        const Rect& b = wbox[w];
        if (g.size() == 1) {
            const float ar = float(b.w) / float(b.h);
            if (b.h >= P.icon_min && b.h <= P.icon_max && ar > 0.6f && ar < 1.7f && comps[g[0]].fill() > 0.18f) {
                pictos.push_back(g[0]);
                continue;
            }
        }
        Element e;
        e.bbox = b;
        e.kind = ElementKind::Text;
        e.atoms = g;
        text_word_ids.push_back(uint32_t(out.words.size()));
        text_word_boxes.push_back(b);
        out.words.push_back(std::move(e));
    }

    // Pictogram fragments -> icons (union of boxes within icon_merge_gap, capped at icon_max).
    {
        std::vector<uint32_t> par(pictos.size());
        std::iota(par.begin(), par.end(), 0u);
        std::vector<Rect> gb(pictos.size());
        for (size_t a = 0; a < pictos.size(); ++a) gb[a] = boxes[pictos[a]];
        auto find = [&](uint32_t i) { while (par[i] != i) i = par[i] = par[par[i]]; return i; };
        for (size_t a = 0; a < pictos.size(); ++a)
            for (size_t b = a + 1; b < pictos.size(); ++b) {
                const uint32_t ra = find(uint32_t(a)), rb = find(uint32_t(b));
                if (ra == rb || gb[ra].inflate(P.icon_merge_gap).inter_area(gb[rb]) == 0) continue;
                const Rect u = gb[ra].unite(gb[rb]);
                if (u.w > P.icon_max || u.h > P.icon_max) continue;
                par[std::max(ra, rb)] = std::min(ra, rb);
                gb[std::min(ra, rb)] = u;
            }
        std::vector<int32_t> slot(pictos.size(), -1);
        std::vector<Element> icons;
        for (size_t a = 0; a < pictos.size(); ++a) {
            const uint32_t r = find(uint32_t(a));
            if (slot[r] < 0) slot[r] = int32_t(icons.size()), icons.emplace_back(), icons.back().kind = ElementKind::Icon;
            Element& e = icons[size_t(slot[r])];
            e.bbox = e.bbox.unite(boxes[pictos[a]]);
            e.atoms.push_back(pictos[a]);
        }
        for (auto& e : icons)
            if (detail::icon_is_salient(frame, luma, e.bbox, P)) out.elements.push_back(std::move(e));
    }

    // Lines = words merged with a wider gap (same geometric rules on word boxes).
    std::vector<uint32_t> widx(text_word_boxes.size());
    std::iota(widx.begin(), widx.end(), 0u);
    std::vector<int> word_ink(text_word_boxes.size());
    for (size_t k = 0; k < text_word_boxes.size(); ++k) {
        std::vector<int> v;
        for (uint32_t a : out.words[text_word_ids[k]].atoms) v.push_back(ink_of(a));
        std::nth_element(v.begin(), v.begin() + ptrdiff_t(v.size() / 2), v.end());
        word_ink[k] = v[v.size() / 2];
    }
    auto word_veto = [&](uint32_t a, uint32_t b) {
        if (P.luma_veto && std::abs(word_ink[a] - word_ink[b]) > P.luma_delta) return true;
        const Rect &A = text_word_boxes[a], &B = text_word_boxes[b];
        const int32_t gap = std::max(A.x, B.x) - std::min(A.right(), B.right());
        return P.gutter_veto && float(gap) > P.gutter_min_gap * float(std::max(A.h, B.h)) &&
               detail::vertical_gutter(mask, A, B, P.gutter_reach);
    };
    for (auto& g : detail::group_line_atoms(text_word_boxes, widx, P.line_gap, P, word_veto)) {
        Element e;
        e.kind = ElementKind::Text;
        for (uint32_t k : g) {
            const Element& w = out.words[text_word_ids[k]];
            e.bbox = e.bbox.unite(w.bbox);
            e.atoms.insert(e.atoms.end(), w.atoms.begin(), w.atoms.end());
        }
        std::sort(e.atoms.begin(), e.atoms.end(), [&](uint32_t a, uint32_t b) { return boxes[a].x < boxes[b].x; });
        out.elements.push_back(std::move(e));
    }

    // Paragraph blocks: stack lines whose vertical gap <= block_leading * height, similar heights, and
    // left-aligned or horizontally overlapping by most of the narrower line.
    {
        std::vector<uint32_t> li;
        for (uint32_t i = 0; i < out.elements.size(); ++i)
            if (out.elements[i].kind == ElementKind::Text) li.push_back(i);
        std::sort(li.begin(), li.end(), [&](uint32_t a, uint32_t b) {
            const Rect &A = out.elements[a].bbox, &B = out.elements[b].bbox;
            return A.y != B.y ? A.y < B.y : A.x < B.x;
        });
        std::vector<uint32_t> par(li.size());
        std::iota(par.begin(), par.end(), 0u);
        auto find = [&](uint32_t i) { while (par[i] != i) i = par[i] = par[par[i]]; return i; };
        for (size_t a = 0; a < li.size(); ++a) {
            const Rect& A = out.elements[li[a]].bbox;
            for (size_t b = a + 1; b < li.size(); ++b) {
                const Rect& B = out.elements[li[b]].bbox;
                if (float(B.y - A.bottom()) > P.block_leading * float(A.h)) break;  // sorted by y
                if (B.y < A.bottom() - A.h / 3) continue;                           // same row, not stacked
                const int32_t hmin = std::min(A.h, B.h), hmax = std::max(A.h, B.h);
                if (float(hmax) > 1.5f * float(hmin)) continue;
                const bool aligned = std::abs(A.x - B.x) <= hmin;
                const bool overlap = A.x_overlap(B) >= std::min(A.w, B.w) * 3 / 4;
                if (!aligned && !overlap) continue;
                const uint32_t ra = find(uint32_t(a)), rb = find(uint32_t(b));
                if (ra != rb) par[std::max(ra, rb)] = std::min(ra, rb);
            }
        }
        std::vector<int32_t> slot(li.size(), -1);
        for (size_t a = 0; a < li.size(); ++a) {
            const uint32_t r = find(uint32_t(a));
            if (slot[r] < 0) slot[r] = int32_t(out.blocks.size()), out.blocks.emplace_back();
            Element& e = out.blocks[size_t(slot[r])];
            e.bbox = e.bbox.unite(out.elements[li[a]].bbox);
            e.atoms.insert(e.atoms.end(), out.elements[li[a]].atoms.begin(), out.elements[li[a]].atoms.end());
        }
    }

    std::vector<Rect> eb(out.elements.size());
    for (size_t i = 0; i < eb.size(); ++i) eb[i] = out.elements[i].bbox;
    out.forest = build_containment_forest(eb);
    for (size_t i = 0; i < eb.size(); ++i) {
        out.elements[i].parent = out.forest.parent[i];
        out.elements[i].depth = out.forest.depth[i];
    }
    return out;
}

}  // namespace dks

