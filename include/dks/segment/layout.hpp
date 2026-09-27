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
