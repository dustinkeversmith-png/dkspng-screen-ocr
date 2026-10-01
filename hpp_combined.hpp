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
        out.elements = read_boxes(out.layout, to_luma(frame));
        return out;
    }

    // OCR pass over an existing layout (used by the pipeline's TextReadStage).
    std::vector<ReadElement> read_boxes(const Layout& layout, const Gray8& gray) const {
        std::vector<ReadElement> elements(layout.elements.size());
        auto work = [&](size_t t) {
            const ocr::Recognizer rec(cls_[t], rp_);
            // Static striping: box i always lands on thread i % T, so each thread's memo sees the
            // same boxes frame after frame (dynamic scheduling scatters them and misses the cache).
            for (size_t i = t; i < elements.size(); i += cls_.size()) {
                ReadElement& re = elements[i];
                const Element& e = layout.elements[i];
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
        return elements;
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

// === C:\Users\Cutie Magic 500\projects\creative\screen-ocr\include\dks\detection\detection_pack.hpp ===
#pragma once
// Pluggable shape / widget classification packs.
//
//   IShapeClassifier   one pack = one classification concern ("widget type", "checkbox state", ...)
//   TemplatePack       exemplar k-NN over ShapeDescriptors; trainable from labelled crops, saved as .dkpk
//   PackRegistry       ordered, mutable set of packs (add / remove / replace at runtime)
//   DetectionStage     pipeline stage: runs every registered pack over the elements it applies to and
//                      attaches {pack.tag_key(), match.tag} SemanticTags
//
// A pack decides what it looks at through applies(): element kinds + size limits, and optionally a
// required tag written by an earlier pack (e.g. a state pack only runs on boxes tagged ui.checkbox).
// The core layout / OCR code never sees any of this.
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "../pipeline/pipeline_stage.hpp"
#include "shape_descriptor.hpp"

namespace dks::detect {

struct ShapeMatch {
    std::string tag;         // e.g. "ui.checkbox", "checked", "icon.search", "math.integral"
    float confidence = 0.f;  // 0..1
    uint32_t class_id = 0;
};

// What a pack gets to see for one element.
struct Patch {
    GrayView luma;                         // element crop (with a small margin)
    Rect box;                              // element box in frame coordinates
    ElementKind kind = ElementKind::Icon;
    const std::vector<SemanticTag>* tags = nullptr;  // tags written so far (may be null)

    const SemanticTag* tag(const std::string& key) const {
        if (!tags) return nullptr;
        for (const auto& t : *tags)
            if (t.key == key) return &t;
        return nullptr;
    }
};

class IShapeClassifier {
public:
    virtual ~IShapeClassifier() = default;
    virtual std::string name() const = 0;     // unique pack name
    virtual std::string tag_key() const = 0;  // SemanticTag key the pack writes
    virtual bool applies(const Patch& p) const = 0;
    virtual std::vector<ShapeMatch> classify(const Patch& p) const = 0;
};

// Applicability filter shared by the built-in packs.
struct PackScope {
    std::vector<ElementKind> kinds{ElementKind::Icon, ElementKind::Container, ElementKind::Image};
    int min_side = 8, max_side = 160;
    std::string require_key;               // only run when this tag key exists ...
    std::vector<std::string> require_any;  // ... with one of these values (empty = any value)

    bool ok(const Patch& p) const {
        if (std::find(kinds.begin(), kinds.end(), p.kind) == kinds.end()) return false;
        if (std::min(p.box.w, p.box.h) < min_side || std::max(p.box.w, p.box.h) > max_side) return false;
        if (!require_key.empty()) {
            const SemanticTag* t = p.tag(require_key);
            if (!t) return false;
            if (!require_any.empty() && std::find(require_any.begin(), require_any.end(), t->value) == require_any.end())
                return false;
        }
        return true;
    }
};

// ------------------------------------------------------------------------------------ TemplatePack
// Exemplar k-NN. Deterministic: distance ties break by exemplar index, vote ties by class id.
class TemplatePack : public IShapeClassifier {
public:
    TemplatePack(std::string name, std::string tag_key, PackScope scope = {}, int k = 5, ShapeWeights w = {})
        : name_(std::move(name)), key_(std::move(tag_key)), scope_(std::move(scope)), k_(k), w_(w) {}

    std::string name() const override { return name_; }
    std::string tag_key() const override { return key_; }
    PackScope& scope() noexcept { return scope_; }
    const std::vector<std::string>& classes() const noexcept { return classes_; }
    size_t size() const noexcept { return ex_.size(); }

    uint32_t class_id(const std::string& label) {
        for (uint32_t i = 0; i < classes_.size(); ++i)
            if (classes_[i] == label) return i;
        classes_.push_back(label);
        return uint32_t(classes_.size() - 1);
    }
    void add_example(const std::string& label, GrayView crop) { ex_.push_back({describe(crop), class_id(label)}); }
    void add_example(const std::string& label, const ShapeDescriptor& d) { ex_.push_back({d, class_id(label)}); }

    bool applies(const Patch& p) const override { return !ex_.empty() && scope_.ok(p); }

    std::vector<ShapeMatch> classify(const Patch& p) const override { return classify(describe(p.luma)); }

    std::vector<ShapeMatch> classify(const ShapeDescriptor& q) const {
        std::vector<std::pair<float, uint32_t>> best;  // (distance, exemplar)
        best.reserve(size_t(k_) + 1);
        for (uint32_t i = 0; i < ex_.size(); ++i) {
            const float d = distance(q, ex_[i].d, w_);
            if (int(best.size()) == k_ && d >= best.back().first) continue;
            best.emplace_back(d, i);
            std::sort(best.begin(), best.end());
            if (int(best.size()) > k_) best.pop_back();
        }
        // Distance-weighted vote.
        std::vector<float> vote(classes_.size(), 0.f);
        float total = 0.f;
        for (const auto& [d, i] : best) {
            const float wgt = 1.f / (1.f + d);
            vote[ex_[i].cls] += wgt;
            total += wgt;
        }
        std::vector<ShapeMatch> out;
        for (uint32_t c = 0; c < vote.size(); ++c)
            if (vote[c] > 0.f) out.push_back({classes_[c], total > 0 ? vote[c] / total : 0.f, c});
        std::sort(out.begin(), out.end(), [](const ShapeMatch& a, const ShapeMatch& b) {
            return a.confidence != b.confidence ? a.confidence > b.confidence : a.class_id < b.class_id;
        });
        return out;
    }

    // ".dkpk": "DKPK" u32 ver=1 | name | tag_key | u32 k | u32 n_classes {str} | u32 n {u16 cls, f32 aspect, f32 fill, 192 u8}
    bool save(const std::string& path) const {
        FILE* f = std::fopen(path.c_str(), "wb");
        if (!f) return false;
        auto u32 = [&](uint32_t v) { std::fwrite(&v, 4, 1, f); };
        auto str = [&](const std::string& s) { u32(uint32_t(s.size())); std::fwrite(s.data(), 1, s.size(), f); };
        std::fwrite("DKPK", 1, 4, f);
        u32(1);
        str(name_);
        str(key_);
        u32(uint32_t(k_));
        u32(uint32_t(classes_.size()));
        for (const auto& c : classes_) str(c);
        u32(uint32_t(ex_.size()));
        for (const auto& e : ex_) {
            const uint16_t c = uint16_t(e.cls);
            std::fwrite(&c, 2, 1, f);
            std::fwrite(&e.d.log_aspect, 4, 1, f);
            std::fwrite(&e.d.fill, 4, 1, f);
            std::fwrite(e.d.hog.data(), 1, kHog, f);
            std::fwrite(e.d.ink.data(), 1, kInk, f);
        }
        return std::fclose(f) == 0;
    }

    static std::shared_ptr<TemplatePack> load(const std::string& path, PackScope scope = {}) {
        FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) return nullptr;
        bool ok = true;
        auto u32 = [&]() { uint32_t v = 0; ok = ok && std::fread(&v, 4, 1, f) == 1; return v; };
        auto str = [&]() {
            const uint32_t n = u32();
            std::string s(ok ? n : 0, '\0');
            ok = ok && n < (1u << 20) && std::fread(s.data(), 1, n, f) == n;
            return s;
        };
        char magic[4] = {};
        ok = std::fread(magic, 1, 4, f) == 4 && std::memcmp(magic, "DKPK", 4) == 0 && u32() == 1;
        std::shared_ptr<TemplatePack> p;
        if (ok) {
            const std::string name = str(), key = str();
            const int k = int(u32());
            p = std::make_shared<TemplatePack>(name, key, std::move(scope), k);
            const uint32_t nc = u32();
            for (uint32_t i = 0; ok && i < nc; ++i) p->classes_.push_back(str());
            const uint32_t n = u32();
            for (uint32_t i = 0; ok && i < n; ++i) {
                Exemplar e;
                uint16_t c = 0;
                ok = std::fread(&c, 2, 1, f) == 1 && std::fread(&e.d.log_aspect, 4, 1, f) == 1 &&
                     std::fread(&e.d.fill, 4, 1, f) == 1 && std::fread(e.d.hog.data(), 1, kHog, f) == size_t(kHog) &&
                     std::fread(e.d.ink.data(), 1, kInk, f) == size_t(kInk);
                e.cls = c;
                if (ok) p->ex_.push_back(e);
            }
        }
        std::fclose(f);
        return ok ? p : nullptr;
    }

private:
    struct Exemplar {
        ShapeDescriptor d;
        uint32_t cls = 0;
    };
    std::string name_, key_;
    PackScope scope_;
    int k_;
    ShapeWeights w_;
    std::vector<std::string> classes_;
    std::vector<Exemplar> ex_;
};

// ------------------------------------------------------------------------------------ registry + stage
class PackRegistry {
public:
    // Adds or replaces (same name) a pack; packs run in insertion order, so a pack that requires a
    // tag must come after the pack that writes it.
    void add(std::shared_ptr<const IShapeClassifier> pack) {
        remove(pack->name());
        packs_.push_back(std::move(pack));
    }
    bool remove(const std::string& name) {
        const auto n = packs_.size();
        packs_.erase(std::remove_if(packs_.begin(), packs_.end(), [&](const auto& p) { return p->name() == name; }),
                     packs_.end());
        return packs_.size() != n;
    }
    const std::vector<std::shared_ptr<const IShapeClassifier>>& packs() const noexcept { return packs_; }

private:
    std::vector<std::shared_ptr<const IShapeClassifier>> packs_;
};

class DetectionStage : public IPipelineStage {
public:
    explicit DetectionStage(std::shared_ptr<const PackRegistry> registry, int margin = 2, float min_confidence = 0.5f)
        : reg_(std::move(registry)), margin_(margin), min_conf_(min_confidence) {}
    const char* name() const override { return "detection"; }

    void process(AnalysisContext& ctx) const override {
        if (!ctx.has_layout) return;
        const Gray8& luma = ctx.ensure_luma();
        if (ctx.tags.size() < ctx.layout.elements.size()) ctx.tags.resize(ctx.layout.elements.size());
        for (size_t i = 0; i < ctx.layout.elements.size(); ++i) {
            const Element& e = ctx.layout.elements[i];
            Patch p;
            p.box = e.bbox;
            p.kind = e.kind;
            p.luma = luma.cview().sub(e.bbox.inflate(margin_).clip(luma.width(), luma.height()));
            for (const auto& pack : reg_->packs()) {
                p.tags = &ctx.tags[i];  // re-read: earlier packs may have added tags
                if (!pack->applies(p)) continue;
                const auto m = pack->classify(p);
                if (!m.empty() && m.front().confidence >= min_conf_)
                    ctx.tag(i, pack->tag_key(), m.front().tag, m.front().confidence, pack->name());
            }
        }
    }

private:
    std::shared_ptr<const PackRegistry> reg_;
    int margin_;
    float min_conf_;
};

}  // namespace dks::detect

// === C:\Users\Cutie Magic 500\projects\creative\screen-ocr\include\dks\detection\rule_packs.hpp ===
#pragma once
// Training-free packs: deterministic rules for specific widgets.
#include <algorithm>
#include <cstdlib>

#include "../segment/ccl.hpp"
#include "detection_pack.hpp"

namespace dks::detect {

// Checkbox / radio state from the control's interior.
//   1. background = patch border median; control = largest component of |luma - bg| > delta
//   2. interior  = central `inner` fraction of the control's bbox
//   3. checked   <=> the interior contains a mark: its luma spread (p95 - p5) exceeds `delta`.
//      An unchecked control has a uniform interior (page background or its own fill colour); a tick,
//      a radio dot or a filled box with a tick all create contrast *inside* the interior.
class CheckStatePack : public IShapeClassifier {
public:
    explicit CheckStatePack(PackScope scope = {}, int delta = 40, float inner = 0.6f)
        : scope_(std::move(scope)), delta_(delta), inner_(inner) {}
    std::string name() const override { return "check_state.rule"; }
    std::string tag_key() const override { return "widget.state"; }
    bool applies(const Patch& p) const override { return scope_.ok(p); }

    std::vector<ShapeMatch> classify(const Patch& p) const override {
        const int spread = interior_spread(p.luma);
        if (spread < 0) return {};
        const bool checked = spread > delta_;
        // Confidence grows with the distance from the decision threshold.
        const float conf = std::clamp(0.5f + float(std::abs(spread - delta_)) / 120.f, 0.5f, 1.f);
        return {{checked ? "checked" : "unchecked", conf, checked ? 0u : 1u}};
    }

    // p95 - p5 luma inside the control's interior; -1 when no control is found.
    int interior_spread(GrayView g) const {
        if (g.width < 4 || g.height < 4) return -1;
        const int bg = border_median(g);
        Gray8 mask(g.width, g.height, 0);
        for (int32_t y = 0; y < g.height; ++y)
            for (int32_t x = 0; x < g.width; ++x) mask.at(x, y) = std::abs(int(g.at(x, y)) - bg) > delta_;
        const LabelResult L = label_components(mask.cview(), Connectivity::Eight);
        Rect b{0, 0, g.width, g.height};  // tight crops: the control is the whole patch
        if (!L.components.empty()) {
            const Component* big = &L.components[0];
            for (const auto& c : L.components)
                if (c.bbox.area() > big->bbox.area()) big = &c;
            if (big->bbox.w >= g.width / 3 && big->bbox.h >= g.height / 3) b = big->bbox;
        }
        const int32_t iw = std::max<int32_t>(1, int32_t(float(b.w) * inner_)), ih = std::max<int32_t>(1, int32_t(float(b.h) * inner_));
        const Rect in{b.x + (b.w - iw) / 2, b.y + (b.h - ih) / 2, iw, ih};
        int hist[256] = {};
        int n = 0;
        for (int32_t y = in.y; y < in.bottom(); ++y)
            for (int32_t x = in.x; x < in.right(); ++x) ++hist[g.at(x, y)], ++n;
        auto pct = [&](int q) {
            int acc = 0;
            for (int v = 0; v < 256; ++v)
                if ((acc += hist[v]) * 100 >= q * n) return v;
            return 255;
        };
        return n ? pct(95) - pct(5) : -1;
    }

private:
    PackScope scope_;
    int delta_;
    float inner_;
};

}  // namespace dks::detect

// === C:\Users\Cutie Magic 500\projects\creative\screen-ocr\include\dks\detection\shape_descriptor.hpp ===
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

// === C:\Users\Cutie Magic 500\projects\creative\screen-ocr\include\dks\eval\latex_metrics.hpp ===
#pragma once
// LaTeX comparison for im2latex-style token strings.
//
// Both sides are canonicalised first, removing notation that does not change the rendered symbols or
// their 2D structure: spacing (\, \; \quad ~ \hspace{..}), sizing (\left \right \big ...), style / font
// switches (\bf \mathrm \cal \displaystyle ...), synonyms (\le -> \leq, \to -> \rightarrow, \lbrack -> [),
// dot spellings (\dots \ldots \cdots ". . .") and braces around a single token ("{ x }" -> "x").
//
//   token edit distance / |gt|, exact match      structure + symbols
//   symbol-bag precision / recall / F1           symbols only (ignores { } ^ _ \frac \sqrt ...)
#include <algorithm>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace dks::eval {

inline std::vector<std::string> latex_canonical(const std::string& s) {
    static const std::set<std::string> drop = {
        "\\,", "\\;", "\\:", "\\!", "\\quad", "\\qquad", "~", "\\", "\\ ", "\\displaystyle", "\\textstyle", "\\scriptstyle",
        "\\scriptscriptstyle", "\\nonumber", "\\big", "\\Big", "\\bigg", "\\Bigg", "\\bigl", "\\bigr", "\\Bigl", "\\Bigr",
        "\\biggl", "\\biggr", "\\Biggl", "\\Biggr", "\\left", "\\right", "\\left.", "\\right.", "\\limits", "\\nolimits",
        "\\bf", "\\rm", "\\it", "\\cal", "\\mathrm", "\\mathcal", "\\mathbf", "\\mathit", "\\mathsf", "\\sf", "\\boldmath",
        "\\operatorname", "\\operatorname*", "\\textrm", "\\mbox", "\\hbox", "\\text", "\\mathtt", "\\tt", "\\emph", "\\mit",
        "\\tiny", "\\small", "\\scriptsize", "\\footnotesize", "\\protect"};
    static const std::map<std::string, std::string> syn = {
        {"\\left(", "("}, {"\\right)", ")"}, {"\\left[", "["}, {"\\right]", "]"}, {"\\left\\{", "\\{"}, {"\\right\\}", "\\}"},
        {"\\left|", "|"}, {"\\right|", "|"}, {"\\left<", "\\langle"}, {"\\right>", "\\rangle"}, {"\\left\\langle", "\\langle"},
        {"\\right\\rangle", "\\rangle"}, {"\\left\\|", "\\Vert"}, {"\\right\\|", "\\Vert"}, {"\\|", "\\Vert"},
        {"\\left\\vert", "|"}, {"\\right\\vert", "|"}, {"\\vert", "|"}, {"\\mid", "|"}, {"\\lbrack", "["}, {"\\rbrack", "]"},
        {"\\lbrace", "\\{"}, {"\\rbrace", "\\}"}, {"\\le", "\\leq"}, {"\\ge", "\\geq"}, {"\\ne", "\\neq"},
        {"\\to", "\\rightarrow"}, {"\\dots", "\\cdots"}, {"\\ldots", "\\cdots"}, {"\\prime", "'"}, {"\\ast", "*"},
        {"\\dag", "\\dagger"}, {"\\sp", "^"}, {"\\sb", "_"}, {"\\over", "\\frac"}, {"\\lgroup", "("}, {"\\rgroup", ")"}};
    std::vector<std::string> t;
    std::istringstream in(s);
    std::string w;
    std::vector<std::string> raw;
    while (in >> w) raw.push_back(w);
    for (size_t i = 0; i < raw.size(); ++i) {
        std::string x = raw[i];
        if (x == "\\hspace" || x == "\\vspace" || x == "\\kern") {  // drop the command and its argument group
            if (i + 1 < raw.size() && raw[i + 1] == "{") {
                int d = 0;
                for (++i; i < raw.size(); ++i) {
                    d += raw[i] == "{" ? 1 : raw[i] == "}" ? -1 : 0;
                    if (d == 0) break;
                }
            }
            continue;
        }
        const auto it = syn.find(x);
        if (it != syn.end()) x = it->second;
        if (drop.count(x)) continue;
        t.push_back(x);
    }
    // ". . ." -> \cdots
    std::vector<std::string> u;
    for (size_t i = 0; i < t.size(); ++i) {
        if (i + 2 < t.size() && t[i] == "." && t[i + 1] == "." && t[i + 2] == ".") {
            u.push_back("\\cdots");
            i += 2;
        } else {
            u.push_back(t[i]);
        }
    }
    // "{ X }" -> "X" and "{ }" -> nothing, until stable
    for (bool changed = true; changed;) {
        changed = false;
        std::vector<std::string> v;
        for (size_t i = 0; i < u.size(); ++i) {
            const bool keep_group = i > 0 && (u[i - 1] == "\\frac" || u[i - 1] == "\\sqrt" || (i > 1 && u[i - 2] == "\\frac"));
            if (u[i] == "{" && i + 1 < u.size() && u[i + 1] == "}" && !keep_group) { ++i; changed = true; continue; }
            if (u[i] == "{" && i + 2 < u.size() && u[i + 2] == "}" && u[i + 1] != "{" && u[i + 1] != "}") {
                v.push_back(u[i + 1]);
                i += 2;
                changed = true;
                continue;
            }
            v.push_back(u[i]);
        }
        u.swap(v);
    }
    return u;
}

inline bool latex_structural(const std::string& t) {
    return t == "{" || t == "}" || t == "^" || t == "_" || t == "\\frac" || t == "\\sqrt" || t == "\\overline" ||
           t == "\\underline" || t == "\\hat" || t == "\\bar" || t == "\\tilde" || t == "\\vec" || t == "\\dot" ||
           t == "\\ddot" || t == "\\widetilde" || t == "\\widehat" || t == "&" || t == "\\\\";
}

struct LatexScore {
    size_t n = 0, exact = 0, edits = 0, tokens = 0;
    size_t sym_tp = 0, sym_pred = 0, sym_gt = 0;

    void add(const std::string& gt, const std::string& pred) {
        const auto g = latex_canonical(gt), p = latex_canonical(pred);
        ++n;
        tokens += g.size();
        // token-level Levenshtein
        std::vector<size_t> prev(p.size() + 1), cur(p.size() + 1);
        for (size_t j = 0; j <= p.size(); ++j) prev[j] = j;
        for (size_t i = 1; i <= g.size(); ++i) {
            cur[0] = i;
            for (size_t j = 1; j <= p.size(); ++j)
                cur[j] = std::min({prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (g[i - 1] == p[j - 1] ? 0 : 1)});
            std::swap(prev, cur);
        }
        edits += prev[p.size()];
        exact += prev[p.size()] == 0;
        std::map<std::string, int> bag;
        for (const auto& x : g)
            if (!latex_structural(x)) ++bag[x], ++sym_gt;
        for (const auto& x : p)
            if (!latex_structural(x)) {
                ++sym_pred;
                if (bag[x] > 0) --bag[x], ++sym_tp;
            }
    }
    double ted() const { return tokens ? double(edits) / double(tokens) : 0; }
    double em() const { return n ? double(exact) / double(n) : 0; }
    double sym_p() const { return sym_pred ? double(sym_tp) / double(sym_pred) : 0; }
    double sym_r() const { return sym_gt ? double(sym_tp) / double(sym_gt) : 0; }
    double sym_f1() const { const double p = sym_p(), r = sym_r(); return p + r > 0 ? 2 * p * r / (p + r) : 0; }
};

}  // namespace dks::eval

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

// === C:\Users\Cutie Magic 500\projects\creative\screen-ocr\include\dks\latex\latex_stage.hpp ===
#pragma once
// LatexStage: the LaTeX character + structure stage as a separable pipeline step.
//
// It is independent of the base text stage: it has its own atlas (math_fonts.dksa), its own
// segmentation (no shared baseline) and its own parser. It considers a box only when
//   * the box is a text candidate (layout Text) that the text stage did not verify as one-baseline
//     text, or `claim_all_text` is set (pages known to be math); and
//   * the math evidence passes (see MathEvidence / LatexStageParams): enough symbols, good template
//     matches, few rare symbols, and — unless claim_all_text — genuine math content.
// Claimed boxes get {"latex", "<tokens>"}; nothing else about the element changes.
#include <memory>
#include <numeric>
#include <string>

#include "../pipeline/pipeline_stage.hpp"
#include "spatial_tree_parsing.hpp"

namespace dks::latex {

struct LatexStageParams {
    bool claim_all_text = false;  // math-only pages: read every Text box as math
    // Gating chosen on im2latex (positives) vs Zenodo desktops + stock wallpapers (negatives), see
    // bench_latex_stage: ~27% of formula images claimed at <= 0.1 false claims per screen.
    int min_symbols = 8;
    float max_mean_dist = 20.f;   // mean template distance of the glyph symbols
    float max_rare = 0.1f;        // share of symbols outside the common math vocabulary
    int min_math_marks = 3;       // structure (^ _ \frac \sqrt) + math-only symbols (Greek, relations, big ops)
    int max_height = 400;
    // Formula regions: the layout splits a formula at scripts / fraction parts (its line rules want one
    // baseline), so text boxes are grown by these fractions of their height and overlapping ones merged.
    float grow_x = 0.8f, grow_y = 0.5f;
    int max_region_members = 24;
};

// Evidence that a crop is a formula.
struct MathEvidence {
    int symbols = 0;
    float mean_dist = 0;
    float rare = 0;      // share of glyphs with a prior penalty (rare in math) or outside the vocabulary
    int math_marks = 0;  // scripts / fractions / radicals + math-only symbols (Greek, operators, relations)
    std::string tex;
};

inline MathEvidence math_evidence(const MathReader& reader, GrayView crop, const ParseParams& pp = {}) {
    MathEvidence ev;
    const auto syms = reader.symbols(crop);
    ev.symbols = int(syms.size());
    if (syms.empty()) return ev;
    float d = 0;
    int n = 0, rare = 0;
    for (const auto& s : syms) {
        if (s.kind == MathSymbol::Kind::Radical) ++ev.math_marks;
        if (s.kind != MathSymbol::Kind::Glyph) continue;
        d += s.dist, ++n;
        if (MathReader::prior(s.ch) >= 5.f) ++rare;
        const SymClass k = class_of(s.ch);
        if ((s.ch >= 0x0391 && s.ch <= 0x03F5) || k == SymClass::BigOp || k == SymClass::Relation) ++ev.math_marks;
    }
    ev.mean_dist = n ? d / float(n) : 99.f;
    ev.rare = n ? float(rare) / float(n) : 1.f;
    ev.tex = parse_latex(syms, pp);
    for (const char* m : {"^ {", "_ {", "\\frac", "\\sqrt"})
        for (size_t p = ev.tex.find(m); p != std::string::npos; p = ev.tex.find(m, p + 1)) ++ev.math_marks;
    return ev;
}

class LatexStage : public IPipelineStage {
public:
    LatexStage(std::shared_ptr<const ocr::Classifier> math_classifier, LatexStageParams p = {}, MathReadParams mp = {},
               ParseParams pp = {}, MathPrior prior = {})
        : cls_(std::move(math_classifier)), reader_(*cls_, mp, std::move(prior)), p_(p), pp_(pp) {}
    const char* name() const override { return "latex"; }

    // Candidate test on the text stage's reading of the box (if it ran).
    bool candidate(const AnalysisContext& ctx, size_t i) const {
        const Element& e = ctx.layout.elements[i];
        if (e.kind != ElementKind::Text || e.bbox.h > p_.max_height) return false;
        if (p_.claim_all_text || i >= ctx.reads.size()) return true;
        const auto& r = ctx.reads[i];
        return !(r.is_text && r.detail.line_fit && r.detail.fit_residual < 0.25f);  // one-baseline text: not ours
    }

    bool accept(const MathEvidence& ev) const {
        if (p_.claim_all_text) return ev.symbols >= 2;  // caller asserts the content is math
        if (ev.symbols < p_.min_symbols || ev.mean_dist > p_.max_mean_dist || ev.rare > p_.max_rare) return false;
        return ev.math_marks >= p_.min_math_marks;
    }

    // Candidate formula regions: grown text boxes merged by overlap. A region is a candidate when at
    // least one member is a candidate (not verified one-baseline text).
    std::vector<Region> regions(const AnalysisContext& ctx) const {
        const auto& E = ctx.layout.elements;
        std::vector<uint32_t> ids;
        std::vector<Rect> grown;
        for (uint32_t i = 0; i < E.size(); ++i) {
            if (E[i].kind != ElementKind::Text || E[i].bbox.h > p_.max_height) continue;
            const Rect& b = E[i].bbox;
            const int32_t gx = int32_t(p_.grow_x * float(b.h)), gy = int32_t(p_.grow_y * float(b.h));
            ids.push_back(i);
            grown.push_back(Rect{b.x - gx, b.y - gy, b.w + 2 * gx, b.h + 2 * gy});
        }
        if (ids.empty()) return {};
        std::vector<uint32_t> par(ids.size());
        std::iota(par.begin(), par.end(), 0u);
        auto find = [&](uint32_t i) { while (par[i] != i) i = par[i] = par[par[i]]; return i; };
        for (size_t a = 0; a < ids.size(); ++a)
            for (size_t b = a + 1; b < ids.size(); ++b)
                if (grown[a].inter_area(grown[b]) > 0) {
                    const uint32_t x = find(uint32_t(a)), y = find(uint32_t(b));
                    if (x != y) par[std::max(x, y)] = std::min(x, y);
                }
        std::vector<Region> out;
        std::vector<int32_t> slot(ids.size(), -1);
        for (size_t a = 0; a < ids.size(); ++a) {
            const uint32_t r = find(uint32_t(a));
            if (slot[r] < 0) slot[r] = int32_t(out.size()), out.emplace_back();
            Region& g = out[size_t(slot[r])];
            g.members.push_back(ids[a]);
            g.box = g.box.unite(E[ids[a]].bbox);
        }
        std::vector<Region> keep;
        for (auto& g : out) {
            if (int(g.members.size()) > p_.max_region_members) continue;  // a whole text column, not a formula
            bool any = false;
            for (uint32_t m : g.members) any |= candidate(ctx, m);
            if (!any) continue;
            // include scripts / limits that sit just outside the member boxes
            int32_t h = 0;
            for (uint32_t m : g.members) h = std::max(h, E[m].bbox.h);
            g.box = g.box.inflate(std::max<int32_t>(2, h * 3 / 10)).clip(ctx.frame.width, ctx.frame.height);
            keep.push_back(std::move(g));
        }
        return keep;
    }

    void process(AnalysisContext& ctx) const override {
        if (!ctx.has_layout) return;
        const Gray8& luma = ctx.ensure_luma();
        for (Region& g : regions(ctx)) {
            const MathEvidence ev = math_evidence(reader_, luma.cview().sub(g.box), pp_);
            if (!accept(ev)) continue;
            const float score = std::clamp(1.f - ev.mean_dist / (2.f * p_.max_mean_dist), 0.f, 1.f);
            g.tags.push_back(SemanticTag{"latex", ev.tex, score, name()});
            for (uint32_t m : g.members) ctx.tag(m, "latex.region", std::to_string(ctx.regions.size()), score, name());
            ctx.regions.push_back(std::move(g));
        }
    }

    const MathReader& reader() const noexcept { return reader_; }
    LatexStageParams& params() noexcept { return p_; }

private:
    std::shared_ptr<const ocr::Classifier> cls_;
    MathReader reader_;
    LatexStageParams p_;
    ParseParams pp_;
};

}  // namespace dks::latex

// === C:\Users\Cutie Magic 500\projects\creative\screen-ocr\include\dks\latex\math_reader.hpp ===
#pragma once
// MathReader: symbol-level reading of a formula crop (Layer 2 of the LaTeX stage).
//
// Unlike the text recogniser there is no shared-baseline assumption: every symbol is found on its own
// (connected components, with stacked parts merged: i j = ≤ ≥ ≡ : ; ! ÷), classified against the
// math atlas, and given a *scale* (cap height implied by its best template) and a *baseline*. Two
// structural symbols are found without templates:
//   Bar      horizontal rule: minus / fraction bar / overline (decided by the parser from context)
//   Radical  √ with its vinculum: a hook on the left plus a bar running over the radicand
#include <algorithm>
#include <cmath>
#include <numeric>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "../ocr/ocr.hpp"
#include "math_symbols.hpp"

namespace dks::latex {

struct MathSymbol {
    enum class Kind : uint8_t { Glyph, Bar, Radical };
    Kind kind = Kind::Glyph;
    Rect box;
    char32_t ch = 0;
    std::string tex;
    float dist = 0;              // template distance (Glyph)
    float scale = 0;             // implied cap height in px (0 = unknown)
    float baseline = 0;          // implied baseline y in px
    int32_t radicand_x = 0;      // Radical: x where the vinculum starts
    ocr::Candidate cand[4];
    int ncand = 0;
};

struct MathReadParams {
    InkParams ink;
    int min_area = 2;
    float bar_aspect = 3.f;      // w / h for a horizontal rule
    float bar_fill = 0.7f;
    float merge_overlap = 0.5f;  // x-overlap / narrower width for stacked parts
    int topk = 4;
    bool rescore = true;         // pass 2: re-rank candidates by prior + implied scale / baseline
    float scale_weight = 14.f;   // distance units per |log| scale mismatch
    float place_weight = 10.f;   // distance units per cap height of punctuation placement error
    float prior_weight = 10.f;   // with a unigram prior loaded: max penalty for the rarest symbol
};

// Unigram symbol prior (label codepoint -> count), e.g. data/math_prior.tsv written by
// `bench_latex --make-prior` from a *training / validation* split. Penalty = w * (1 - log(c+1)/log(max+1)).
struct MathPrior {
    std::unordered_map<char32_t, float> penalty;
    float unseen = 1.f;
    bool empty() const noexcept { return penalty.empty(); }

    static MathPrior from_counts(const std::unordered_map<char32_t, double>& counts) {
        MathPrior p;
        double mx = 1;
        for (const auto& [c, n] : counts) mx = std::max(mx, n);
        const double lm = std::log(mx + 1);
        for (const auto& [c, n] : counts) p.penalty[c] = float(1.0 - std::log(n + 1) / lm);
        return p;
    }
    bool load(const std::string& path) {
        std::ifstream in(path);
        if (!in) return false;
        std::unordered_map<char32_t, double> counts;
        unsigned long cp;
        double n;
        while (in >> cp >> n) counts[char32_t(cp)] = n;
        *this = from_counts(counts);
        return !empty();
    }
    float operator()(char32_t c) const {
        const auto it = penalty.find(c);
        return it == penalty.end() ? unseen : it->second;
    }
};

class MathReader {
public:
    MathReader(const ocr::Classifier& cls, MathReadParams p = {}, MathPrior prior = {})
        : cls_(cls), p_(p), prior_(std::move(prior)) {}

    std::vector<MathSymbol> symbols(GrayView crop) const {
        std::vector<MathSymbol> out;
        if (crop.width < 3 || crop.height < 3) return out;
        const ocr::CropContext C = ocr::analyze_crop(crop, p_.ink);
        const auto& comps = C.ccl.components;
        std::vector<uint32_t> keep;
        for (uint32_t i = 0; i < comps.size(); ++i)
            if (comps[i].area >= p_.min_area) keep.push_back(i);
        if (keep.empty()) return out;

        std::vector<int32_t> hs;
        for (uint32_t i : keep) hs.push_back(comps[i].bbox.h);
        std::nth_element(hs.begin(), hs.begin() + ptrdiff_t(hs.size() / 2), hs.end());
        const float Hmed = float(std::max(1, hs[hs.size() / 2]));

        auto is_bar = [&](const Component& c) {
            return float(c.bbox.w) >= p_.bar_aspect * float(c.bbox.h) && c.bbox.w >= 4 && c.fill() >= p_.bar_fill;
        };
        std::vector<int32_t> radical_start(comps.size(), -1);
        for (uint32_t i : keep) {
            const int32_t vx = radical_vinculum(C, comps[i]);
            if (vx < 0) continue;
            // A real radical covers a radicand: another component centred under the vinculum.
            const Rect& rb = comps[i].bbox;
            for (uint32_t j : keep)
                if (j != i && comps[j].bbox.x + comps[j].bbox.w / 2 >= vx &&
                    rb.contains_point(comps[j].bbox.x + comps[j].bbox.w / 2, comps[j].bbox.y + comps[j].bbox.h / 2)) {
                    radical_start[i] = vx;
                    break;
                }
        }

        // Stacked-part merging (union-find over kept components).
        std::vector<uint32_t> par(comps.size());
        std::iota(par.begin(), par.end(), 0u);
        auto find = [&](uint32_t i) { while (par[i] != i) i = par[i] = par[par[i]]; return i; };
        for (size_t a = 0; a < keep.size(); ++a)
            for (size_t b = a + 1; b < keep.size(); ++b) {
                const Component &A = comps[keep[a]], &B = comps[keep[b]];
                if (radical_start[keep[a]] >= 0 || radical_start[keep[b]] >= 0) continue;
                const Rect &ra = A.bbox, &rb = B.bbox;
                const int32_t xo = ra.x_overlap(rb), yo = ra.y_overlap(rb);
                const int32_t wmin = std::min(ra.w, rb.w);
                if (float(xo) < p_.merge_overlap * float(wmin)) continue;
                const int32_t gap = -yo;  // vertical gap (<= 0 means overlap)
                const bool bar_a = is_bar(A), bar_b = is_bar(B);
                bool merge = false;
                if (bar_a && bar_b) {  // = ≡ : similar-width rules close together
                    merge = gap >= 0 && float(gap) <= 0.6f * Hmed && float(std::max(ra.w, rb.w)) <= 1.25f * float(wmin);
                } else if (bar_a != bar_b) {  // ≤ ≥ ÷-bar: rule no wider than the glyph (a fraction bar is wider)
                    const Rect& bar = bar_a ? ra : rb;
                    const Rect& g = bar_a ? rb : ra;
                    merge = gap >= 0 && float(gap) <= 0.35f * Hmed && float(bar.w) <= 1.15f * float(g.w) && float(g.h) <= 1.2f * Hmed;
                } else {  // dots of i j ! ; : ? — one part tiny, or both tiny
                    const int32_t amin = std::min(A.area, B.area), amax = std::max(A.area, B.area);
                    const bool tiny_pair = float(std::max(ra.h, rb.h)) <= 0.4f * Hmed;
                    merge = gap >= -1 && float(gap) <= 0.45f * Hmed && (float(amin) <= 0.3f * float(amax) || tiny_pair);
                }
                if (merge) {
                    const uint32_t x = find(keep[a]), y = find(keep[b]);
                    if (x != y) par[std::max(x, y)] = std::min(x, y);
                }
            }
        std::vector<std::vector<uint32_t>> groups;
        {
            std::vector<int32_t> slot(comps.size(), -1);
            for (uint32_t i : keep) {
                const uint32_t r = find(i);
                if (slot[r] < 0) slot[r] = int32_t(groups.size()), groups.emplace_back();
                groups[size_t(slot[r])].push_back(i);
            }
        }

        for (const auto& g : groups) {
            MathSymbol s;
            std::vector<ocr::Source> src;
            for (uint32_t c : g) src.push_back({c, INT32_MIN, INT32_MAX}), s.box = s.box.unite(comps[c].bbox);
            if (g.size() == 1 && radical_start[g[0]] >= 0) {
                s.kind = MathSymbol::Kind::Radical;
                s.ch = 0x221A;
                s.tex = "\\sqrt";
                s.radicand_x = radical_start[g[0]];
                out.push_back(std::move(s));
                continue;
            }
            if (g.size() == 1 && is_bar(comps[g[0]])) {
                s.kind = MathSymbol::Kind::Bar;
                s.ch = U'-';
                s.tex = "-";
                // Minus width is ~1.14 cap heights in Computer Modern; its centre sits on the math axis
                // (~0.37 cap heights above the baseline).
                s.scale = float(s.box.w) / 1.14f;
                s.baseline = float(s.box.y) + 0.5f * float(s.box.h) + 0.37f * s.scale;
                out.push_back(std::move(s));
                continue;
            }
            const ocr::GlyphFeature f = ocr::sources_feature(C, src.data(), src.data() + src.size(), &s.box);
            s.ncand = int(cls_.classify(f, s.cand, size_t(std::min(p_.topk, 4))));
            if (s.ncand == 0) continue;
            s.ch = s.cand[0].ch;
            s.tex = tex_of(s.ch);
            s.dist = s.cand[0].dist;
            const ocr::Template& t = cls_.atlas().templates[s.cand[0].tmpl];
            const float rel = std::max(0.15f, t.bot_rel - t.top_rel);
            s.scale = float(s.box.h) / rel;
            s.baseline = float(s.box.bottom()) - t.bot_rel * s.scale;
            out.push_back(std::move(s));
        }
        if (p_.rescore) rescore(out);
        std::sort(out.begin(), out.end(), [](const MathSymbol& a, const MathSymbol& b) {
            return a.box.x != b.box.x ? a.box.x < b.box.x : a.box.y < b.box.y;
        });
        return out;
    }

    // Symbol frequency prior (distance units): letters, digits, common operators free; rare symbols pay.
    static float prior(char32_t c) {
        if ((c >= U'a' && c <= U'z') || (c >= U'A' && c <= U'Z') || (c >= U'0' && c <= U'9')) return 0.f;
        switch (c) {
            case U'(': case U')': case U'[': case U']': case U'+': case U'-': case U'=': case U',': case U'.':
            case U'|': case U'/': case U'\'': return 0.f;
            default: break;
        }
        if (c >= 0x03B1 && c <= 0x03C9) return 1.f;  // Greek
        if (c >= 0x0393 && c <= 0x03A9) return 1.f;
        switch (c) {
            case 0x2211: case 0x220F: case 0x222B: case 0x2202: case 0x221E: case 0x00B1: case 0x00D7: case 0x2264:
            case 0x2265: case 0x2192: case 0x2208: case 0x2261: case 0x2248: case 0x223C: case 0x22C5: case 0x2032:
            case U'<': case U'>': case U'{': case U'}': case U';': case U':': case U'!': case U'*': return 2.f;
            default: return 5.f;
        }
    }

private:
    static bool placement_punct(char32_t c) {
        return c == U',' || c == U'.' || c == U'\'' || c == 0x22C5 || c == 0x00B7 || c == 0x2032 || c == U'`';
    }

    // Pass 2. With the main scale S and baseline B of the formula (median of the larger symbols):
    //   * case twins / 1-l-I: the candidate whose implied scale fits S (on the baseline) or a script
    //     level 0.71 S / 0.5 S (off the baseline) wins;
    //   * punctuation (, . ' ·): placement against B decides;
    //   * a frequency prior keeps rare symbols from winning on near-ties.
    void rescore(std::vector<MathSymbol>& out) const {
        std::vector<float> sc;
        for (const auto& s : out)
            if (s.kind == MathSymbol::Kind::Glyph && s.scale > 0) sc.push_back(s.scale);
        if (sc.size() < 2) return;
        std::sort(sc.begin(), sc.end(), std::greater<float>());
        const float S = sc[(std::max<size_t>(1, (sc.size() * 6 + 9) / 10) - 1) / 2];  // median of the larger 60%
        std::vector<float> bl;
        for (const auto& s : out)
            if (s.kind == MathSymbol::Kind::Glyph && s.scale >= 0.84f * S) bl.push_back(s.baseline);
        if (bl.empty()) return;
        std::sort(bl.begin(), bl.end());
        const float B = bl[bl.size() / 2];
        for (auto& s : out) {
            if (s.kind != MathSymbol::Kind::Glyph || s.ncand == 0) continue;
            float best = 1e30f;
            int bi = 0;
            for (int c = 0; c < s.ncand; ++c) {
                const ocr::Template& t = cls_.atlas().templates[s.cand[c].tmpl];
                const float rel = std::max(0.15f, t.bot_rel - t.top_rel);
                const float sc_c = float(s.box.h) / rel;
                const float bl_c = float(s.box.bottom()) - t.bot_rel * sc_c;
                float pen;
                if (placement_punct(s.cand[c].ch)) {
                    // compare the symbol's top/bottom with where this punctuation sits on the main line
                    pen = (std::fabs(float(s.box.y) - (B + t.top_rel * S)) + std::fabs(float(s.box.bottom()) - (B + t.bot_rel * S))) / S;
                    pen *= p_.place_weight / p_.scale_weight;
                } else if (std::fabs(bl_c - B) <= 0.15f * S) {
                    pen = std::fabs(std::log(sc_c / S));
                } else {
                    pen = std::min(std::fabs(std::log(sc_c / (0.71f * S))), std::fabs(std::log(sc_c / (0.5f * S))));
                }
                const float pr = prior_.empty() ? prior(s.cand[c].ch) : p_.prior_weight * prior_(s.cand[c].ch);
                const float score = s.cand[c].dist + pr + p_.scale_weight * pen;
                if (score < best) best = score, bi = c;
            }
            if (bi != 0) {
                const ocr::Template& t = cls_.atlas().templates[s.cand[bi].tmpl];
                const float rel = std::max(0.15f, t.bot_rel - t.top_rel);
                s.ch = s.cand[bi].ch;
                s.tex = tex_of(s.ch);
                s.dist = s.cand[bi].dist;
                s.scale = float(s.box.h) / rel;
                s.baseline = float(s.box.bottom()) - t.bot_rel * s.scale;
            }
        }
    }

public:

private:
    // A radical is one component: a hook on the left whose ink reaches the bottom quarter, and a
    // horizontal vinculum in the top rows spanning most of the width. Returns the x where the
    // vinculum starts (the radicand's left edge), or -1.
    static int32_t radical_vinculum(const ocr::CropContext& C, const Component& c) {
        const Rect& b = c.bbox;
        if (b.h < 8 || b.w < b.h / 2 || c.fill() > 0.35f) return -1;
        const int32_t band = std::max(1, b.h / 8);
        int32_t best_len = 0, best_x0 = -1;
        bool hook_low = false;
        for (uint32_t i = c.run_begin; i < c.run_end; ++i) {
            const Run& r = C.ccl.runs[i];
            if (r.y < b.y + band && r.x1 - r.x0 > best_len) best_len = r.x1 - r.x0, best_x0 = r.x0;
            if (r.y >= b.bottom() - b.h / 4 && r.x0 < b.x + b.w * 2 / 5) hook_low = true;
        }
        if (!hook_low || float(best_len) < 0.55f * float(b.w) || best_x0 <= b.x) return -1;
        return best_x0;
    }

    const ocr::Classifier& cls_;
    MathReadParams p_;
    MathPrior prior_;
};

}  // namespace dks::latex

// === C:\Users\Cutie Magic 500\projects\creative\screen-ocr\include\dks\latex\math_symbols.hpp ===
#pragma once
// Math atlas labels -> LaTeX. Labels are the codepoints stored in data/math_fonts.dksa
// (tools/build_math_atlas.py): ASCII letters/digits/punctuation, Greek, operators.
#include <string>
#include <unordered_map>

#include "../core/utf8.hpp"

namespace dks::latex {

enum class SymClass : uint8_t { Ordinary, BigOp, Relation, Binary, Open, Close, Punct };

struct SymbolInfo {
    const char* tex;
    SymClass cls;
};

inline const std::unordered_map<char32_t, SymbolInfo>& symbol_table() {
    static const std::unordered_map<char32_t, SymbolInfo> t = [] {
        std::unordered_map<char32_t, SymbolInfo> m;
        const char* greek[] = {"\\alpha", "\\beta", "\\gamma", "\\delta", "\\varepsilon", "\\zeta", "\\eta", "\\theta",
                               "\\iota", "\\kappa", "\\lambda", "\\mu", "\\nu", "\\xi", "o", "\\pi", "\\rho",
                               "\\varsigma", "\\sigma", "\\tau", "\\upsilon", "\\varphi", "\\chi", "\\psi", "\\omega"};
        for (int i = 0; i < 25; ++i) m[char32_t(0x03B1 + i)] = {greek[i], SymClass::Ordinary};
        m[0x03F5] = {"\\epsilon", SymClass::Ordinary};
        m[0x03D1] = {"\\vartheta", SymClass::Ordinary};
        m[0x03D5] = {"\\phi", SymClass::Ordinary};
        m[0x03F1] = {"\\varrho", SymClass::Ordinary};
        m[0x03D6] = {"\\varpi", SymClass::Ordinary};
        const std::pair<char32_t, const char*> upper[] = {{0x0393, "\\Gamma"}, {0x0394, "\\Delta"}, {0x0398, "\\Theta"},
                                                          {0x039B, "\\Lambda"}, {0x039E, "\\Xi"}, {0x03A0, "\\Pi"},
                                                          {0x03A3, "\\Sigma"}, {0x03A5, "\\Upsilon"}, {0x03A6, "\\Phi"},
                                                          {0x03A8, "\\Psi"}, {0x03A9, "\\Omega"}};
        for (auto& [c, s] : upper) m[c] = {s, SymClass::Ordinary};
        const struct { char32_t c; const char* s; SymClass k; } ops[] = {
            {U'+', "+", SymClass::Binary}, {U'-', "-", SymClass::Binary}, {U'=', "=", SymClass::Relation},
            {U'<', "<", SymClass::Relation}, {U'>', ">", SymClass::Relation}, {U'(', "(", SymClass::Open},
            {U')', ")", SymClass::Close}, {U'[', "[", SymClass::Open}, {U']', "]", SymClass::Close},
            {U'{', "\\{", SymClass::Open}, {U'}', "\\}", SymClass::Close}, {U'|', "|", SymClass::Ordinary},
            {U'/', "/", SymClass::Ordinary}, {U',', ",", SymClass::Punct}, {U'.', ".", SymClass::Punct},
            {U';', ";", SymClass::Punct}, {U':', ":", SymClass::Relation}, {U'!', "!", SymClass::Ordinary},
            {U'\'', "'", SymClass::Ordinary}, {U'*', "*", SymClass::Binary}, {U'&', "\\&", SymClass::Ordinary},
            {U'#', "\\#", SymClass::Ordinary}, {U'"', "\"", SymClass::Ordinary}, {U'~', "\\sim", SymClass::Relation},
            {0x00B1, "\\pm", SymClass::Binary}, {0x2213, "\\mp", SymClass::Binary}, {0x00D7, "\\times", SymClass::Binary},
            {0x00F7, "\\div", SymClass::Binary}, {0x00B7, "\\cdot", SymClass::Binary}, {0x22C5, "\\cdot", SymClass::Binary},
            {0x2217, "*", SymClass::Binary}, {0x2218, "\\circ", SymClass::Binary}, {0x2022, "\\bullet", SymClass::Binary},
            {0x22C6, "\\star", SymClass::Binary}, {0x2264, "\\leq", SymClass::Relation}, {0x2265, "\\geq", SymClass::Relation},
            {0x2260, "\\neq", SymClass::Relation}, {0x2248, "\\approx", SymClass::Relation}, {0x2261, "\\equiv", SymClass::Relation},
            {0x223C, "\\sim", SymClass::Relation}, {0x2243, "\\simeq", SymClass::Relation}, {0x2245, "\\cong", SymClass::Relation},
            {0x221D, "\\propto", SymClass::Relation}, {0x226A, "\\ll", SymClass::Relation}, {0x226B, "\\gg", SymClass::Relation},
            {0x221E, "\\infty", SymClass::Ordinary}, {0x2202, "\\partial", SymClass::Ordinary}, {0x2207, "\\nabla", SymClass::Ordinary},
            {0x2211, "\\sum", SymClass::BigOp}, {0x220F, "\\prod", SymClass::BigOp}, {0x222B, "\\int", SymClass::BigOp},
            {0x222E, "\\oint", SymClass::BigOp}, {0x221A, "\\sqrt", SymClass::Ordinary}, {0x2192, "\\rightarrow", SymClass::Relation},
            {0x2190, "\\leftarrow", SymClass::Relation}, {0x2194, "\\leftrightarrow", SymClass::Relation},
            {0x21D2, "\\Rightarrow", SymClass::Relation}, {0x21D4, "\\Leftrightarrow", SymClass::Relation},
            {0x27F6, "\\longrightarrow", SymClass::Relation}, {0x21A6, "\\mapsto", SymClass::Relation},
            {0x2191, "\\uparrow", SymClass::Relation}, {0x2193, "\\downarrow", SymClass::Relation}, {0x2208, "\\in", SymClass::Relation},
            {0x2209, "\\notin", SymClass::Relation}, {0x2282, "\\subset", SymClass::Relation}, {0x2283, "\\supset", SymClass::Relation},
            {0x2286, "\\subseteq", SymClass::Relation}, {0x2287, "\\supseteq", SymClass::Relation}, {0x222A, "\\cup", SymClass::Binary},
            {0x2229, "\\cap", SymClass::Binary}, {0x2200, "\\forall", SymClass::Ordinary}, {0x2203, "\\exists", SymClass::Ordinary},
            {0x00AC, "\\neg", SymClass::Ordinary}, {0x2227, "\\wedge", SymClass::Binary}, {0x2228, "\\vee", SymClass::Binary},
            {0x2297, "\\otimes", SymClass::Binary}, {0x2295, "\\oplus", SymClass::Binary}, {0x2299, "\\odot", SymClass::Binary},
            {0x2020, "\\dagger", SymClass::Ordinary}, {0x2021, "\\ddagger", SymClass::Ordinary}, {0x2016, "\\Vert", SymClass::Ordinary},
            {0x27E8, "\\langle", SymClass::Open}, {0x27E9, "\\rangle", SymClass::Close}, {0x210F, "\\hbar", SymClass::Ordinary},
            {0x2113, "\\ell", SymClass::Ordinary}, {0x2026, "\\ldots", SymClass::Ordinary}, {0x22EF, "\\cdots", SymClass::Ordinary},
            {0x22EE, "\\vdots", SymClass::Ordinary}, {0x2032, "\\prime", SymClass::Ordinary}, {0x22A5, "\\perp", SymClass::Relation},
            {0x2225, "\\parallel", SymClass::Relation}, {0x25B3, "\\triangle", SymClass::Ordinary}, {0x2111, "\\Im", SymClass::Ordinary},
            {0x211C, "\\Re", SymClass::Ordinary}, {0x2118, "\\wp", SymClass::Ordinary}, {0x2135, "\\aleph", SymClass::Ordinary},
            {0x266F, "\\sharp", SymClass::Ordinary}, {0x266D, "\\flat", SymClass::Ordinary}, {0x2294, "\\sqcup", SymClass::Binary},
            {0x2293, "\\sqcap", SymClass::Binary}, {0x22C4, "\\diamond", SymClass::Binary}, {0x21C0, "\\rightharpoonup", SymClass::Relation},
        };
        for (const auto& o : ops) m[o.c] = {o.s, o.k};
        return m;
    }();
    return t;
}

inline std::string tex_of(char32_t c) {
    const auto& t = symbol_table();
    const auto it = t.find(c);
    if (it != t.end()) return it->second.tex;
    std::string s;
    utf8_append(s, c);
    return s;
}

inline SymClass class_of(char32_t c) {
    const auto& t = symbol_table();
    const auto it = t.find(c);
    return it != t.end() ? it->second.cls : SymClass::Ordinary;
}

}  // namespace dks::latex

// === C:\Users\Cutie Magic 500\projects\creative\screen-ocr\include\dks\latex\spatial_tree_parsing.hpp ===
#pragma once
// Spatial Tree Parsing (deterministic operator dominance) — Layer 3 of the LaTeX stage.
//
// Once math symbols and their boxes are extracted (MathReader), the 2D layout is folded into a tree,
// innermost structures first:
//   1. Radicals   bounding-box containment: every symbol right of the radical's vinculum start and
//                 inside its box is the radicand -> \sqrt { ... }
//   1b. Accents   \bar \tilde \hat \vec \dot: a small mark directly over exactly one glyph
//   2. Fractions  horizontal rules (w/h >= 3, fill >= 0.7), narrowest first: symbols whose x-span lies
//                 within the bar's x-span and sit directly above / below form numerator / denominator ->
//                 \frac { num } { den }. Rule with content only above -> \overline; nothing -> minus.
//   3. Limits     big operators (∑ ∏) take symbols stacked above / below within their x-span.
//   4. Scripts    the remaining nodes are read left to right against the main line (median scale S and
//                 baseline B of the full-size symbols). Each symbol's scale and baseline come from its best
//                 template's metrics, so x-height letters, ascenders and descenders are comparable. A
//                 smaller symbol whose baseline is raised is a superscript, lowered a subscript; runs of
//                 script symbols are parsed recursively (nested scripts, fractions inside scripts).
// Output uses im2latex-style spaced tokens: "x _ { 1 } ^ { 2 }", "\frac { a } { b }".
#include <algorithm>
#include <string>
#include <vector>

#include "math_reader.hpp"

namespace dks::latex {

struct ParseParams {
    float script_scale = 0.84f;   // scale / main scale below this = script size
    float sup_raise = 0.22f;      // baseline raised by > this * S  -> superscript
    float sub_drop = 0.12f;       // baseline lowered by > this * S -> subscript
    float stack_reach = 2.2f;     // numerator / denominator / limits within this * S of the bar / operator
};

namespace detail {

struct Node {
    Rect box;
    std::string tex;
    MathSymbol::Kind kind = MathSymbol::Kind::Glyph;
    char32_t ch = 0;
    float scale = 0, baseline = 0;
    int32_t radicand_x = 0;
    bool group = false;  // composite (fraction / radical / limits): always main line
};

inline float main_scale(const std::vector<Node>& v) {
    std::vector<float> s;
    for (const auto& n : v)
        if (n.scale > 0 && n.kind == MathSymbol::Kind::Glyph) s.push_back(n.scale);
    if (s.empty())
        for (const auto& n : v)
            if (n.scale > 0) s.push_back(n.scale);
    if (s.empty()) return 0;
    std::sort(s.begin(), s.end(), std::greater<float>());
    const size_t k = std::max<size_t>(1, (s.size() * 6 + 9) / 10);  // median of the larger 60%
    return s[(k - 1) / 2];
}

inline std::string parse_nodes(std::vector<Node> v, const ParseParams& P, int depth);

inline std::string join(const std::vector<Node>& v, const ParseParams& P, int depth) {
    return parse_nodes(v, P, depth + 1);
}

inline bool centre_within(const Rect& r, const Rect& span, int slack) {
    const int32_t c = r.x + r.w / 2;
    return c >= span.x - slack && c < span.right() + slack;
}

inline std::string parse_nodes(std::vector<Node> v, const ParseParams& P, int depth) {
    if (v.empty() || depth > 12) return "";
    // A sub-expression (script, numerator, denominator, radicand) that is one one-like stem is "1".
    if (depth > 0 && v.size() == 1 && !v[0].group && v[0].kind == MathSymbol::Kind::Glyph &&
        (v[0].ch == U'l' || v[0].ch == U'I' || v[0].ch == U'J' || v[0].ch == 0x131))
        return "1";
    float S = main_scale(v);
    if (S <= 0) S = 10.f;

    // 1. radicals, smallest first
    for (;;) {
        int best = -1;
        for (int i = 0; i < int(v.size()); ++i)
            if (!v[size_t(i)].group && v[size_t(i)].kind == MathSymbol::Kind::Radical &&
                (best < 0 || v[size_t(i)].box.area() < v[size_t(best)].box.area()))
                best = i;
        if (best < 0) break;
        const Node R = v[size_t(best)];
        std::vector<Node> inside, rest;
        for (int i = 0; i < int(v.size()); ++i) {
            if (i == best) continue;
            const Node& n = v[size_t(i)];
            const int32_t cx = n.box.x + n.box.w / 2, cy = n.box.y + n.box.h / 2;
            (cx >= R.radicand_x && R.box.contains_point(cx, cy) ? inside : rest).push_back(n);
        }
        Node g = R;
        g.group = true;
        g.tex = "\\sqrt { " + join(inside, P, depth) + " }";
        for (const auto& n : inside) g.box = g.box.unite(n.box);
        g.scale = S;
        g.baseline = float(R.box.bottom()) - 0.1f * S;
        rest.push_back(g);
        v.swap(rest);
    }

    // 1b. accents: a short rule / tilde / hat / arrow / dot directly above exactly one glyph, with
    //     nothing stacked above it (so a fraction bar over a one-glyph denominator is never taken)
    for (;;) {
        bool changed = false;
        for (size_t a = 0; a < v.size() && !changed; ++a) {
            const Node& acc = v[a];
            if (acc.group) continue;
            const char* cmd = nullptr;
            if (acc.kind == MathSymbol::Kind::Bar) cmd = "\\bar";
            else if (acc.ch == U'~' || acc.ch == 0x223C) cmd = "\\tilde";
            else if (acc.ch == U'^' || acc.ch == 0x02C6 || acc.ch == 0x2227) cmd = "\\hat";
            else if (acc.ch == 0x2192 || acc.ch == 0x21C0) cmd = "\\vec";
            else if (acc.ch == 0x22C5 || acc.ch == 0x00B7 || acc.ch == U'.') cmd = "\\dot";
            if (!cmd || float(acc.box.h) > 0.45f * S) continue;
            int under = -1, count = 0;
            for (size_t b = 0; b < v.size(); ++b) {
                if (b == a) continue;
                const Node& g = v[b];
                if (g.kind != MathSymbol::Kind::Glyph || g.group) continue;
                if (!centre_within(acc.box, g.box, 1) || g.box.y < acc.box.bottom() - 1) continue;
                if (float(g.box.y - acc.box.bottom()) > 0.35f * S) continue;
                if (float(acc.box.w) > 1.6f * float(g.box.w) + 2.f) continue;
                under = int(b), ++count;
            }
            if (count != 1) continue;
            bool above = false;
            for (size_t b = 0; b < v.size() && !above; ++b)
                above = b != a && centre_within(v[b].box, acc.box, 1) && v[b].box.bottom() <= acc.box.y + 1 &&
                        float(acc.box.y - v[b].box.bottom()) <= P.stack_reach * S;
            if (above) continue;
            Node g = v[size_t(under)];
            g.tex = std::string(cmd) + " { " + g.tex + " }";
            g.box = g.box.unite(acc.box);
            std::vector<Node> rest;
            for (size_t b = 0; b < v.size(); ++b)
                if (b != a && b != size_t(under)) rest.push_back(v[b]);
            rest.push_back(g);
            v.swap(rest);
            changed = true;
        }
        if (!changed) break;
    }

    // 2. fraction bars, narrowest first (inner fractions become groups before the outer bar looks)
    for (;;) {
        int best = -1;
        for (int i = 0; i < int(v.size()); ++i) {
            const Node& n = v[size_t(i)];
            if (n.group || n.kind != MathSymbol::Kind::Bar) continue;
            // only bars that actually have something stacked over or under them
            bool stacked = false;
            for (const auto& o : v)
                if (&o != &n && centre_within(o.box, n.box, 1) && o.box.w <= n.box.w + 2 &&
                    (o.box.bottom() <= n.box.y + 1 || o.box.y >= n.box.bottom() - 1) &&
                    std::min(std::abs(o.box.bottom() - n.box.y), std::abs(o.box.y - n.box.bottom())) <= int32_t(P.stack_reach * S))
                    stacked = true;
            if (stacked && (best < 0 || n.box.w < v[size_t(best)].box.w)) best = i;
        }
        if (best < 0) break;
        const Node B = v[size_t(best)];
        std::vector<Node> num, den, rest;
        const int32_t reach = int32_t(P.stack_reach * S * 2.f);
        for (int i = 0; i < int(v.size()); ++i) {
            if (i == best) continue;
            const Node& n = v[size_t(i)];
            const bool in_span = n.box.x >= B.box.x - 2 && n.box.right() <= B.box.right() + 2;
            if (in_span && n.box.bottom() <= B.box.y + 1 && B.box.y - n.box.bottom() <= reach) num.push_back(n);
            else if (in_span && n.box.y >= B.box.bottom() - 1 && n.box.y - B.box.bottom() <= reach) den.push_back(n);
            else rest.push_back(n);
        }
        Node g = B;
        g.group = true;
        if (!num.empty() && !den.empty()) g.tex = "\\frac { " + join(num, P, depth) + " } { " + join(den, P, depth) + " }";
        else if (!num.empty()) g.tex = "\\overline { " + join(num, P, depth) + " }";
        else g.tex = "\\underline { " + join(den, P, depth) + " }";
        for (const auto& n : num) g.box = g.box.unite(n.box);
        for (const auto& n : den) g.box = g.box.unite(n.box);
        g.scale = S;
        g.baseline = float(B.box.y) + 0.5f * float(B.box.h) + 0.37f * S;  // bar sits on the math axis
        rest.push_back(g);
        v.swap(rest);
    }

    // 3. big-operator limits
    for (size_t i = 0; i < v.size(); ++i) {
        Node& op = v[i];
        if (op.group || (op.ch != 0x2211 && op.ch != 0x220F)) continue;
        std::vector<Node> above, below, rest;
        const int32_t reach = int32_t(P.stack_reach * S);
        for (size_t j = 0; j < v.size(); ++j) {
            if (j == i) continue;
            const Node& n = v[j];
            if (centre_within(n.box, op.box, 2) && n.box.bottom() <= op.box.y + 1 && op.box.y - n.box.bottom() <= reach) above.push_back(n);
            else if (centre_within(n.box, op.box, 2) && n.box.y >= op.box.bottom() - 1 && n.box.y - op.box.bottom() <= reach) below.push_back(n);
        }
        if (above.empty() && below.empty()) continue;
        Node g = op;
        g.group = true;
        g.tex = op.tex;
        if (!below.empty()) g.tex += " _ { " + join(below, P, depth) + " }";
        if (!above.empty()) g.tex += " ^ { " + join(above, P, depth) + " }";
        for (const auto& n : above) g.box = g.box.unite(n.box);
        for (const auto& n : below) g.box = g.box.unite(n.box);
        for (size_t j = 0; j < v.size(); ++j) {
            if (j == i) continue;
            bool used = false;
            for (const auto& n : above) used |= n.box == v[j].box;
            for (const auto& n : below) used |= n.box == v[j].box;
            if (!used) rest.push_back(v[j]);
        }
        rest.push_back(g);
        v.swap(rest);
        i = size_t(-1);  // restart: indices changed
    }

    // 4. main line + scripts
    std::sort(v.begin(), v.end(), [](const Node& a, const Node& b) {
        return a.box.x != b.box.x ? a.box.x < b.box.x : a.box.y < b.box.y;
    });
    S = main_scale(v);
    if (S <= 0) S = 10.f;
    std::vector<float> bl;
    for (const auto& n : v)
        if (n.group || n.scale >= P.script_scale * S) bl.push_back(n.baseline);
    std::sort(bl.begin(), bl.end());
    const float B = bl.empty() ? 0.f : bl[bl.size() / 2];

    enum Role { Main, Sup, Sub };
    auto role = [&](const Node& n) {
        if (n.group || n.scale <= 0 || n.scale >= P.script_scale * S) return Main;
        if (n.ch == U'.' || n.ch == U',' || n.ch == U';') return Main;  // punctuation sits on the line
        if (n.baseline < B - P.sup_raise * S) return Sup;
        if (n.baseline > B + P.sub_drop * S) return Sub;
        return Main;
    };
    // "1" context: at 4-5 px a Computer-Modern 1 and a slanted l / I / J are the same stem. A lone
    // one-like glyph that is a whole script group, or that touches a digit on the main line, is "1".
    auto one_like = [](const Node& n) {
        return !n.group && n.kind == MathSymbol::Kind::Glyph && (n.ch == U'l' || n.ch == U'I' || n.ch == U'J' || n.ch == 0x131);
    };
    auto digit = [](const Node& n) { return !n.group && ((n.ch >= U'0' && n.ch <= U'9') || n.tex == "1"); };
    for (size_t i = 0; i < v.size(); ++i) {
        if (!one_like(v[i]) || role(v[i]) != Main) continue;
        for (size_t j : {i - 1, i + 1}) {
            if (j >= v.size() || role(v[j]) != Main || !digit(v[j])) continue;
            const int32_t gap = j < i ? v[i].box.x - v[j].box.right() : v[j].box.x - v[i].box.right();
            if (float(gap) <= 0.35f * S) v[i].tex = "1";
        }
    }
    auto script = [&](const std::vector<Node>& g) {
        if (g.size() == 1 && one_like(g[0])) return std::string("1");
        return join(g, P, depth);
    };

    std::string out;
    auto emit = [&](const std::string& t) {
        if (t.empty()) return;
        if (!out.empty()) out += ' ';
        out += t;
    };
    for (size_t i = 0; i < v.size();) {
        if (role(v[i]) == Main) {
            emit(v[i].tex);
            ++i;
        } else if (out.empty()) {
            emit("{ }");  // pre-script with no base
        }
        std::vector<Node> sup, sub;
        while (i < v.size() && role(v[i]) != Main) (role(v[i]) == Sup ? sup : sub).push_back(v[i]), ++i;
        if (!sub.empty()) emit("_ { " + script(sub) + " }");
        if (!sup.empty()) emit("^ { " + script(sup) + " }");
    }
    return out;
}

}  // namespace detail

inline std::string parse_latex(const std::vector<MathSymbol>& syms, const ParseParams& P = {}) {
    std::vector<detail::Node> v;
    v.reserve(syms.size());
    for (const auto& s : syms) {
        detail::Node n;
        n.box = s.box;
        n.tex = s.tex;
        n.kind = s.kind;
        n.ch = s.ch;
        n.scale = s.scale;
        n.baseline = s.baseline;
        n.radicand_x = s.radicand_x;
        v.push_back(std::move(n));
    }
    return detail::parse_nodes(std::move(v), P, 0);
}

}  // namespace dks::latex

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

// === C:\Users\Cutie Magic 500\projects\creative\screen-ocr\include\dks\pipeline\core_stages.hpp ===
#pragma once
// The baseline behaviour as pipeline stages: cursor masking, layout (Layer 0) and the OCR pass over
// every box with text verification (Layer 2). A pipeline of LayoutStage + TextReadStage produces
// exactly what ScreenReader::read() produces.
#include <functional>
#include <memory>

#include "pipeline_stage.hpp"

namespace dks {

// Removes a known overlay rectangle (typically the mouse cursor, see win32::cursor_rect) before
// segmentation: each row inside the rect is replaced by a linear blend of the pixels just left and
// right of it, so the pointer can no longer bridge glyph atoms. The rect is also recorded as an
// exclusion for later stages.
class CursorMaskStage : public IPipelineStage {
public:
    using RectProvider = std::function<bool(Rect&)>;
    explicit CursorMaskStage(RectProvider provider) : provider_(std::move(provider)) {}
    const char* name() const override { return "cursor_mask"; }

    void process(AnalysisContext& ctx) const override {
        Rect r;
        if (!provider_ || !provider_(r)) return;
        r = r.clip(ctx.frame.width, ctx.frame.height);
        if (r.empty()) return;
        ctx.exclusions.push_back(r);
        uint8_t* px = ctx.writable_pixels();
        const int bpp = bytes_per_pixel(ctx.frame.format);
        const ptrdiff_t stride = ctx.frame.stride_bytes;
        const int32_t xl = std::max(0, r.x - 1), xr = std::min(ctx.frame.width - 1, r.right());
        for (int32_t y = r.y; y < r.bottom(); ++y) {
            uint8_t* row = px + stride * y;
            for (int32_t x = r.x; x < r.right(); ++x) {
                const int32_t span = std::max(1, xr - xl);
                const int32_t t = x - xl;
                for (int c = 0; c < bpp; ++c)
                    row[x * bpp + c] = uint8_t((int(row[xl * bpp + c]) * (span - t) + int(row[xr * bpp + c]) * t) / span);
            }
        }
    }

private:
    RectProvider provider_;
};

class LayoutStage : public IPipelineStage {
public:
    explicit LayoutStage(LayoutParams p = {}) : p_(p) {}
    const char* name() const override { return "layout"; }
    void process(AnalysisContext& ctx) const override {
        ctx.layout = analyze_layout(ctx.frame, p_);
        ctx.has_layout = true;
        ctx.tags.assign(ctx.layout.elements.size(), {});
        for (size_t i = 0; i < ctx.layout.elements.size(); ++i)
            ctx.tag(i, "kind", kind_name(ctx.layout.elements[i].kind), 1.f, name());
    }

private:
    LayoutParams p_;
};

// OCR of every box + verification (the baseline ScreenReader behaviour). Tags verified boxes with
// {"text", reading}.
class TextReadStage : public IPipelineStage {
public:
    explicit TextReadStage(std::shared_ptr<const ScreenReader> reader) : reader_(std::move(reader)) {}
    const char* name() const override { return "text"; }
    void process(AnalysisContext& ctx) const override {
        if (!ctx.has_layout) return;
        ctx.reads = reader_->read_boxes(ctx.layout, ctx.ensure_luma());
        for (size_t i = 0; i < ctx.reads.size(); ++i)
            if (ctx.reads[i].is_text) ctx.tag(i, "text", ctx.reads[i].text, ctx.reads[i].confidence, name());
    }

private:
    std::shared_ptr<const ScreenReader> reader_;
};

}  // namespace dks

// === C:\Users\Cutie Magic 500\projects\creative\screen-ocr\include\dks\pipeline\pipeline_stage.hpp ===
#pragma once
// Multi-pass pipeline blackboard.
//
//   Layer 0  core segmentation      LayoutStage              (edge map, run CCL, layout, hierarchy)
//   Layer 1  detection packs        CursorMaskStage, DetectionStage (pluggable IShapeClassifier packs)
//   Layer 2  recognisers            TextReadStage (ui_fonts.dksa), LatexStage (math_fonts.dksa)
//   Layer 3  structural aggregators LaTeX spatial tree, ...
//
// Stages communicate only through AnalysisContext: the core layout and OCR types are never extended;
// extra meaning is attached as SemanticTags per element. Stages can be added, removed, reordered or
// switched off at runtime; the core engines are unchanged and still usable on their own.
#include <algorithm>
#include <chrono>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "../pipeline.hpp"

namespace dks {

// Open-ended per-element annotation, e.g. {"widget", "ui.checkbox"}, {"widget.state", "checked"},
// {"latex", "\\frac { a } { b }"}, {"text", "Save"}.
struct SemanticTag {
    std::string key;
    std::string value;
    float score = 1.f;
    std::string source;  // stage / pack that produced it
};

// A derived area produced by a stage (e.g. a formula spanning several layout elements).
struct Region {
    Rect box;
    std::vector<uint32_t> members;  // layout element indices it was built from
    std::vector<SemanticTag> tags;
};

struct StageTiming {
    std::string stage;
    double ms = 0;
};

struct AnalysisContext {
    ColorView frame;                     // current pixels (points at `owned` after a stage rewrote them)
    std::vector<uint8_t> owned;          // private BGRA/RGBA copy, made on first write
    Gray8 luma;                          // see ensure_luma()
    bool has_luma = false;
    Layout layout;                       // Layer 0 output
    bool has_layout = false;
    std::vector<ReadElement> reads;      // Layer 2 OCR output (same order as layout.elements), may be empty
    std::vector<std::vector<SemanticTag>> tags;  // tags[element index]
    std::vector<Region> regions;         // derived areas (Layer 3 aggregators)
    std::vector<Rect> exclusions;        // regions to ignore (cursor, overlays)
    std::vector<StageTiming> timings;

    explicit AnalysisContext(const ColorView& f) : frame(f) {}

    // Copy-on-write access to the pixels (e.g. cursor inpainting); invalidates luma.
    uint8_t* writable_pixels() {
        if (owned.empty()) {
            const size_t row_bytes = size_t(frame.width) * size_t(bytes_per_pixel(frame.format));
            owned.resize(row_bytes * size_t(frame.height));
            for (int32_t y = 0; y < frame.height; ++y) std::memcpy(owned.data() + row_bytes * size_t(y), frame.row(y), row_bytes);
            frame = ColorView{owned.data(), frame.width, frame.height, ptrdiff_t(row_bytes), frame.format};
        }
        has_luma = false;
        return owned.data();
    }

    const Gray8& ensure_luma() {
        if (!has_luma) luma = to_luma(frame), has_luma = true;
        return luma;
    }

    void tag(size_t element, std::string key, std::string value, float score, std::string source) {
        if (tags.size() < layout.elements.size()) tags.resize(layout.elements.size());
        tags[element].push_back(SemanticTag{std::move(key), std::move(value), score, std::move(source)});
    }

    const SemanticTag* find_tag(size_t element, const std::string& key) const {
        if (element >= tags.size()) return nullptr;
        const SemanticTag* best = nullptr;
        for (const auto& t : tags[element])
            if (t.key == key && (!best || t.score > best->score)) best = &t;
        return best;
    }
};

// Base pipeline step.
class IPipelineStage {
public:
    virtual ~IPipelineStage() = default;
    virtual const char* name() const = 0;
    virtual void process(AnalysisContext& ctx) const = 0;
};

class Pipeline {
public:
    // Appends (or inserts before `before`) a stage. Stage names must be unique.
    Pipeline& add(std::shared_ptr<IPipelineStage> stage, const std::string& before = "") {
        remove(stage->name());
        auto it = std::find_if(stages_.begin(), stages_.end(), [&](const Entry& e) { return e.stage->name() == before; });
        stages_.insert(it, Entry{std::move(stage), true});
        return *this;
    }
    bool remove(const std::string& name) {
        const auto n = stages_.size();
        stages_.erase(std::remove_if(stages_.begin(), stages_.end(), [&](const Entry& e) { return e.stage->name() == name; }),
                      stages_.end());
        return stages_.size() != n;
    }
    bool set_enabled(const std::string& name, bool on) {
        for (auto& e : stages_)
            if (e.stage->name() == name) return e.enabled = on, true;
        return false;
    }
    std::vector<std::string> names() const {
        std::vector<std::string> v;
        for (const auto& e : stages_) v.push_back(std::string(e.stage->name()) + (e.enabled ? "" : " (off)"));
        return v;
    }

    AnalysisContext run(const ColorView& frame) const {
        AnalysisContext ctx(frame);
        for (const auto& e : stages_) {
            if (!e.enabled) continue;
            const auto t0 = std::chrono::steady_clock::now();
            e.stage->process(ctx);
            ctx.timings.push_back({e.stage->name(),
                                   std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count()});
        }
        return ctx;
    }

private:
    struct Entry {
        std::shared_ptr<IPipelineStage> stage;
        bool enabled;
    };
    std::vector<Entry> stages_;
};

}  // namespace dks

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

// Current mouse cursor rectangle in capture_screen() coordinates (virtual-screen origin). GDI BitBlt
// normally does not include the hardware cursor, but software cursors, pointer trails and remote
// sessions do; CursorMaskStage uses this rect to inpaint it before segmentation.
inline bool cursor_rect(Rect& out) {
    CURSORINFO ci{};
    ci.cbSize = sizeof(ci);
    if (!GetCursorInfo(&ci) || !(ci.flags & CURSOR_SHOWING)) return false;
    int32_t w = 32, h = 32, hx = 0, hy = 0;
    ICONINFO ii{};
    if (GetIconInfo(ci.hCursor, &ii)) {
        hx = int32_t(ii.xHotspot), hy = int32_t(ii.yHotspot);
        BITMAP bm{};
        if (ii.hbmMask && GetObject(ii.hbmMask, sizeof(bm), &bm)) {
            w = bm.bmWidth;
            h = ii.hbmColor ? bm.bmHeight : bm.bmHeight / 2;  // monochrome cursors stack AND/XOR masks
        }
        if (ii.hbmMask) DeleteObject(ii.hbmMask);
        if (ii.hbmColor) DeleteObject(ii.hbmColor);
    }
    out = Rect{ci.ptScreenPos.x - hx - GetSystemMetrics(SM_XVIRTUALSCREEN), ci.ptScreenPos.y - hy - GetSystemMetrics(SM_YVIRTUALSCREEN),
               w, h};
    return true;
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

