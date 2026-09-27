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
