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
