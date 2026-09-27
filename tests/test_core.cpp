// Minimal self-checking tests for the deterministic primitives (no framework needed).
#include <cstdio>
#include <string>

#include "dks/dks.hpp"
#include "dks/ocr/features.hpp"

using namespace dks;

static int failures = 0;
#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) { std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); ++failures; } \
    } while (0)

static Gray8 from_ascii(const char* const* rows, int h) {
    const int w = int(std::string(rows[0]).size());
    Gray8 g(w, h, 0);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) g.at(x, y) = rows[y][x] == '#' ? 1 : 0;
    return g;
}

int main() {
    // --- geometry
    const Rect a{0, 0, 10, 10}, b{5, 5, 10, 10};
    CHECK(a.inter_area(b) == 25);
    CHECK(std::abs(a.iou(b) - 25.f / 175.f) < 1e-6f);
    CHECK(Rect({0, 0, 100, 100}).contains(Rect{10, 10, 5, 5}));
    CHECK(!a.contains(b));

    // --- CCL: 8- vs 4-connectivity, diagonal touch
    const char* diag[] = {"#..", ".#.", "..#"};
    const Gray8 d = from_ascii(diag, 3);
    CHECK(label_components(d.cview(), Connectivity::Eight).components.size() == 1);
    CHECK(label_components(d.cview(), Connectivity::Four).components.size() == 3);

    // U shape: two arms joined only at the bottom row must merge (union across rows)
    const char* u[] = {"#...#", "#...#", "#####"};
    const LabelResult lu = label_components(from_ascii(u, 3).cview());
    CHECK(lu.components.size() == 1);
    CHECK(lu.components[0].area == 9);
    CHECK((lu.components[0].bbox == Rect{0, 0, 5, 3}));

    // --- Euler number / holes / components
    const char* ring[] = {"#####", "#...#", "#...#", "#####"};             // 'o': 1 component, 1 hole
    const char* eight[] = {"###", "#.#", "###", "#.#", "###"};           // 'B'/'8': 2 holes
    const char* colon[] = {"#", ".", ".", "#"};                          // ':' : 2 components
    CHECK(ocr::euler_number(from_ascii(ring, 4).cview()) == 0);
    CHECK(ocr::euler_number(from_ascii(eight, 5).cview()) == -1);
    CHECK(ocr::euler_number(from_ascii(colon, 4).cview()) == 2);
    CHECK(ocr::euler_number(d.cview()) == 1);  // diagonal line is one 8-connected component

    // --- containment forest
    const std::vector<Rect> boxes = {{0, 0, 100, 100}, {10, 10, 50, 50}, {20, 20, 5, 5}, {200, 0, 10, 10}};
    const Forest f = build_containment_forest(boxes);
    CHECK(f.parent[0] == -1 && f.parent[1] == 0 && f.parent[2] == 1 && f.parent[3] == -1);
    CHECK(f.depth[2] == 2);

    // --- metrics
    CHECK(eval::levenshtein(U"kitten", U"sitting") == 3);
    eval::TextScore ts;
    ts.add("Hello", "hello");
    CHECK(ts.edits == 1 && ts.edits_ci == 0 && ts.exact_ci == 1);
    CHECK(utf8_encode(utf8_decode("€£©")) == "€£©");

    // --- feature determinism: identical input -> identical descriptor
    const Gray8 r = from_ascii(ring, 4);
    Gray8 ink(r.width(), r.height());
    for (int y = 0; y < r.height(); ++y)
        for (int x = 0; x < r.width(); ++x) ink.at(x, y) = r.at(x, y) ? 255 : 0;
    const auto f1 = ocr::make_feature(ink.cview(), r.cview()), f2 = ocr::make_feature(ink.cview(), r.cview());
    CHECK(f1.bmp == f2.bmp && f1.holes == 1 && f1.ncomp == 1);
    CHECK(ocr::distance(f1, f2) == 0.f);
    CHECK(ocr::lower_bound(f1, f2) <= ocr::distance(f1, f2));

    std::printf(failures ? "%d FAILURES\n" : "all tests passed\n", failures);
    return failures ? 1 : 0;
}
