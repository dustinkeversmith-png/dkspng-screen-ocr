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
