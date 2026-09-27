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
