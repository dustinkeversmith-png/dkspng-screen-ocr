// Tests for the modular layer: LaTeX spatial parser, LaTeX canonicalisation, pack registry +
// DetectionStage chaining, Pipeline stage management, and (when the atlas is present) equivalence of
// the pipeline's layout+text stages with ScreenReader::read(). No downloaded data needed.
#include <cstdio>
#include <fstream>
#include <string>

#include "dks/detection/detection_pack.hpp"
#include "dks/eval/latex_metrics.hpp"
#include "dks/latex/spatial_tree_parsing.hpp"
#include "dks/pipeline/core_stages.hpp"

using namespace dks;

static int failures = 0;
#define CHECK(cond)                                                                                  \
    do {                                                                                             \
        if (!(cond)) { std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); ++failures; } \
    } while (0)
#define CHECK_EQ(a, b)                                                                                              \
    do {                                                                                                            \
        const std::string _a = (a), _b = (b);                                                                       \
        if (_a != _b) { std::printf("FAIL %s:%d\n   got      %s\n   expected %s\n", __FILE__, __LINE__, _a.c_str(), _b.c_str()); ++failures; } \
    } while (0)

// Synthetic symbol: box, label; scale = cap height, baseline = y of the glyph's baseline.
static latex::MathSymbol sym(int x, int y, int w, int h, char32_t ch, float scale, float baseline) {
    latex::MathSymbol s;
    s.box = Rect{x, y, w, h};
    s.ch = ch;
    s.tex = latex::tex_of(ch);
    s.scale = scale;
    s.baseline = baseline;
    return s;
}
static latex::MathSymbol bar(int x, int y, int w) {
    latex::MathSymbol s;
    s.kind = latex::MathSymbol::Kind::Bar;
    s.box = Rect{x, y, w, 1};
    s.ch = U'-';
    s.tex = "-";
    s.scale = float(w) / 1.14f;
    s.baseline = float(y) + 0.5f + 0.37f * s.scale;
    return s;
}

static void test_parser() {
    // x^2 + y_i = 1   (main size 10, baseline y=20)
    std::vector<latex::MathSymbol> v = {
        sym(0, 13, 6, 7, U'x', 10, 20),  sym(7, 8, 4, 6, U'2', 6, 14),     sym(14, 12, 7, 7, U'+', 10, 20),
        sym(24, 13, 6, 9, U'y', 10, 20), sym(31, 19, 3, 5, U'i', 6, 24),   sym(38, 14, 7, 4, U'=', 10, 20),
        sym(48, 10, 4, 10, U'1', 10, 20)};
    CHECK_EQ(latex::parse_latex(v), "x ^ { 2 } + y _ { i } = 1");

    // \frac { a } { b }
    v = {sym(2, 2, 6, 7, U'a', 10, 9), bar(0, 11, 10), sym(2, 13, 6, 10, U'b', 10, 23)};
    CHECK_EQ(latex::parse_latex(v), "\\frac { a } { b }");

    // minus between two symbols stays a minus
    v = {sym(0, 13, 6, 7, U'a', 10, 20), bar(8, 16, 8), sym(18, 10, 6, 10, U'b', 10, 20)};
    CHECK_EQ(latex::parse_latex(v), "a - b");

    // \sqrt { x }
    latex::MathSymbol r;
    r.kind = latex::MathSymbol::Kind::Radical;
    r.box = Rect{0, 5, 16, 16};
    r.radicand_x = 6;
    r.tex = "\\sqrt";
    v = {r, sym(8, 12, 6, 7, U'x', 10, 19)};
    CHECK_EQ(latex::parse_latex(v), "\\sqrt { x }");

    // \sum _ { i } ^ { n }
    v = {sym(0, 10, 14, 14, 0x2211, 14, 24), sym(4, 0, 5, 6, U'n', 6, 6), sym(5, 26, 3, 6, U'i', 6, 32)};
    CHECK_EQ(latex::parse_latex(v), "\\sum _ { i } ^ { n }");

    // a lone one-like stem in a subscript reads as 1
    v = {sym(0, 13, 6, 7, U'x', 10, 20), sym(7, 19, 3, 6, U'l', 6, 24)};
    CHECK_EQ(latex::parse_latex(v), "x _ { 1 }");
}

static void test_latex_canonical() {
    eval::LatexScore s;
    s.add("\\left( x \\right) ^ { 2 } \\, \\le { y }", "( x ) ^ 2 \\leq y");
    CHECK(s.exact == 1);
    eval::LatexScore t;
    t.add("\\frac { 1 } { 2 }", "\\frac { 1 } { 3 }");
    CHECK(t.exact == 0 && t.edits == 1);
}

// Dummy packs to exercise registry order and tag chaining.
struct TypePack : detect::IShapeClassifier {
    std::string name() const override { return "type"; }
    std::string tag_key() const override { return "widget"; }
    bool applies(const detect::Patch&) const override { return true; }
    std::vector<detect::ShapeMatch> classify(const detect::Patch& p) const override {
        return {{p.box.w > p.box.h ? "ui.switch" : "ui.checkbox", 0.9f, 0}};
    }
};
struct StatePack : detect::IShapeClassifier {
    std::string name() const override { return "state"; }
    std::string tag_key() const override { return "widget.state"; }
    bool applies(const detect::Patch& p) const override {
        const auto* t = p.tag("widget");
        return t && t->value == "ui.checkbox";
    }
    std::vector<detect::ShapeMatch> classify(const detect::Patch&) const override { return {{"checked", 0.8f, 0}}; }
};

static void test_packs_and_pipeline() {
    // Frame: white page with a square box and a wide box (both hollow outlines).
    const int W = 120, H = 60;
    std::vector<uint8_t> px(size_t(W) * H * 4, 255);
    auto rect = [&](int x0, int y0, int w, int h) {
        for (int y = y0; y < y0 + h; ++y)
            for (int x = x0; x < x0 + w; ++x)
                if (y == y0 || y == y0 + h - 1 || x == x0 || x == x0 + w - 1)
                    for (int c = 0; c < 3; ++c) px[(size_t(y) * W + x) * 4 + c] = 0;
    };
    rect(10, 10, 30, 30);
    rect(60, 20, 50, 20);
    const ColorView frame{px.data(), W, H, W * 4, PixelFormat::BGRA32};

    auto reg = std::make_shared<detect::PackRegistry>();
    reg->add(std::make_shared<TypePack>());
    reg->add(std::make_shared<StatePack>());
    CHECK(reg->packs().size() == 2);
    reg->add(std::make_shared<StatePack>());  // same name replaces
    CHECK(reg->packs().size() == 2);

    Pipeline pipe;
    pipe.add(std::make_shared<LayoutStage>()).add(std::make_shared<detect::DetectionStage>(reg, 2, 0.5f));
    CHECK(pipe.names().size() == 2);
    AnalysisContext ctx = pipe.run(frame);
    CHECK(ctx.has_layout && !ctx.layout.elements.empty());
    int checkbox = 0, checked = 0, sw = 0;
    for (size_t i = 0; i < ctx.layout.elements.size(); ++i) {
        const auto* t = ctx.find_tag(i, "widget");
        const auto* s = ctx.find_tag(i, "widget.state");
        if (t && t->value == "ui.checkbox") ++checkbox, checked += s != nullptr;
        if (t && t->value == "ui.switch") {
            ++sw;
            CHECK(s == nullptr);  // state pack must not run on switches
        }
    }
    CHECK(checkbox >= 1 && checked == checkbox && sw >= 1);

    // Removing a pack at runtime removes its tags on the next run; disabling a stage skips it.
    reg->remove("state");
    ctx = pipe.run(frame);
    for (size_t i = 0; i < ctx.layout.elements.size(); ++i) CHECK(ctx.find_tag(i, "widget.state") == nullptr);
    pipe.set_enabled("detection", false);
    ctx = pipe.run(frame);
    for (size_t i = 0; i < ctx.layout.elements.size(); ++i) CHECK(ctx.find_tag(i, "widget") == nullptr);
    CHECK(pipe.remove("detection") && pipe.names().size() == 1);

    // Cursor masking inpaints the rect and records an exclusion.
    Pipeline p2;
    p2.add(std::make_shared<CursorMaskStage>([](Rect& r) { r = Rect{5, 5, 8, 8}; return true; }));
    const AnalysisContext c2 = p2.run(frame);
    CHECK(c2.exclusions.size() == 1 && !c2.owned.empty());
    CHECK(px[(size_t(10) * W + 10) * 4] == 0);  // original frame untouched (copy-on-write)
}

static void test_equivalence() {
    ocr::Atlas atlas;
    if (!atlas.load("data/ui_fonts.dksa")) {
        std::printf("  (skip equivalence: data/ui_fonts.dksa not found)\n");
        return;
    }
    atlas.prune(3.0f);
    const ocr::Classifier cls(atlas);
    auto reader = std::make_shared<ScreenReader>(cls);
    // Synthetic frame with bars that look like text lines.
    const int W = 200, H = 60;
    std::vector<uint8_t> px(size_t(W) * H * 4, 255);
    for (int k = 0; k < 12; ++k)
        for (int y = 20; y < 30; ++y)
            for (int x = 10 + k * 14; x < 10 + k * 14 + 3 + (k % 3); ++x)
                for (int c = 0; c < 3; ++c) px[(size_t(y) * W + x) * 4 + c] = 20;
    const ColorView frame{px.data(), W, H, W * 4, PixelFormat::BGRA32};
    const ScreenRead direct = reader->read(frame);
    Pipeline pipe;
    pipe.add(std::make_shared<LayoutStage>()).add(std::make_shared<TextReadStage>(reader));
    const AnalysisContext ctx = pipe.run(frame);
    CHECK(ctx.reads.size() == direct.elements.size());
    for (size_t i = 0; i < ctx.reads.size() && i < direct.elements.size(); ++i) {
        CHECK(ctx.reads[i].text == direct.elements[i].text);
        CHECK(ctx.reads[i].is_text == direct.elements[i].is_text);
        CHECK(ctx.reads[i].element.bbox == direct.elements[i].element.bbox);
    }
}

int main() {
    test_parser();
    test_latex_canonical();
    test_packs_and_pipeline();
    test_equivalence();
    std::printf(failures ? "%d FAILURES\n" : "all pipeline tests passed\n", failures);
    return failures ? 1 : 0;
}
