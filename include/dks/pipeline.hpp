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
#include <atomic>
#include <cstring>
#include <unordered_map>
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
    // Which layout kinds get the OCR pass. Text inside a container is detected and read as its own
    // Text element, so reading the container crop again repeats that work on a multi-line crop that
    // verification almost always rejects. Icons stay on: a readable "icon" is promoted to text.
    bool read_containers = false;
    bool read_images = false;
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
        boxes_.resize(n);
    }

    LayoutParams& layout_params() noexcept { return lp_; }
    TextVerifyParams& verify_params() noexcept { return vp_; }
    unsigned threads() const noexcept { return unsigned(cls_.size()); }
    // Exact caches for repeated content: the glyph memo (per classifier) and the box cache, which maps a
    // box crop's exact pixels to its whole reading, so an unchanged screen region is not re-read.
    void enable_memo(bool on, size_t box_cache_entries = 4096) {
        for (auto& c : cls_) c.enable_memo(on);
        box_cap_ = on ? box_cache_entries : 0;
        for (auto& b : boxes_) b.clear();
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
                if ((e.kind == ElementKind::Container && !vp_.read_containers) || (e.kind == ElementKind::Image && !vp_.read_images))
                    continue;
                const Rect r = e.bbox.inflate(2).clip(gray.width(), gray.height());
                const GrayView crop = gray.cview().sub(r);
                if (box_cap_) {
                    auto& cache = boxes_[t];
                    const uint64_t h = crop_hash(crop);
                    auto it = cache.find(h);
                    if (it != cache.end() && it->second.same(crop)) {
                        ++box_hits_;
                        re.detail = it->second.rec;
                    } else {
                        re.detail = rec.recognize(crop);
                        if (cache.size() >= box_cap_) cache.clear();
                        cache[h] = BoxEntry::make(crop, re.detail);
                    }
                } else {
                    re.detail = rec.recognize(crop);
                }
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

    struct Counters {
        uint64_t classify_calls = 0, search_calls = 0, memo_hits = 0, distance_evals = 0;
        double search_ms = 0;
    };
    Counters counters() const {
        Counters c;
        for (const auto& k : cls_) {
            c.classify_calls += k.classify_calls, c.search_calls += k.search_calls, c.memo_hits += k.memo_hits;
            c.distance_evals += k.distance_evals, c.search_ms += k.search_ms;
        }
        return c;
    }

    // Aggregate classifier counters over all thread copies.
    uint64_t memo_hits() const {
        uint64_t s = 0;
        for (const auto& c : cls_) s += c.memo_hits;
        return s;
    }

    uint64_t box_hits() const noexcept { return box_hits_; }

private:
    struct BoxEntry {
        int32_t w = 0, h = 0;
        std::vector<uint8_t> pix;
        ocr::Recognition rec;
        static BoxEntry make(GrayView c, const ocr::Recognition& r) {
            BoxEntry e;
            e.w = c.width, e.h = c.height, e.rec = r;
            e.pix.resize(size_t(c.width) * size_t(c.height));
            for (int32_t y = 0; y < c.height; ++y) std::memcpy(&e.pix[size_t(y) * size_t(c.width)], c.row(y), size_t(c.width));
            return e;
        }
        bool same(GrayView c) const {
            if (c.width != w || c.height != h) return false;
            for (int32_t y = 0; y < h; ++y)
                if (std::memcmp(&pix[size_t(y) * size_t(w)], c.row(y), size_t(w)) != 0) return false;
            return true;
        }
    };
    static uint64_t crop_hash(GrayView c) {
        uint64_t h = 0x9E3779B97F4A7C15ull ^ (uint64_t(c.width) << 32) ^ uint64_t(c.height);
        for (int32_t y = 0; y < c.height; ++y) {
            const uint8_t* p = c.row(y);
            int32_t x = 0;
            for (; x + 8 <= c.width; x += 8) {
                uint64_t v;
                std::memcpy(&v, p + x, 8);
                h = (h ^ v) * 0x100000001B3ull;
                h ^= h >> 29;
            }
            for (; x < c.width; ++x) h = (h ^ p[x]) * 0x100000001B3ull;
        }
        return h;
    }

    mutable std::vector<ocr::Classifier> cls_;
    mutable std::vector<std::unordered_map<uint64_t, BoxEntry>> boxes_;
    size_t box_cap_ = 0;
    mutable std::atomic<uint64_t> box_hits_{0};
    ocr::RecognizerParams rp_;
    LayoutParams lp_;
    TextVerifyParams vp_;
};

}  // namespace dks
