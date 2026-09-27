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
