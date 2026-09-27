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
