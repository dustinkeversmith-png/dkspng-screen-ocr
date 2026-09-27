// Print the recogniser's glyph decisions for one crop.
//   ocr_debug <image> [x y w h] [--atlas data/ui_fonts.dksa]
#include <cstdio>

#include "common.hpp"
#include "dks/dks.hpp"
#include "dks/ocr/ocr.hpp"

using namespace dks;

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: ocr_debug <image> [x y w h]\n"); return 1; }
    ocr::Atlas atlas;
    std::string err;
    if (!atlas.load(bench::arg_value(argc, argv, "--atlas", "data/ui_fonts.dksa"), &err)) {
        std::fprintf(stderr, "%s\n", err.c_str());
        return 1;
    }
    atlas.prune(std::stof(bench::arg_value(argc, argv, "--prune", "3.0")));
    ocr::Classifier cls(atlas);
    ocr::Recognizer rec(cls);
    rec.params().trace = bench::has_flag(argc, argv, "--trace");
    win32::Frame img = win32::load_image(argv[1]);
    if (img.empty()) { std::fprintf(stderr, "cannot load\n"); return 1; }
    const Gray8 gray = to_luma(img.view());
    Rect r{0, 0, gray.width(), gray.height()};
    if (argc >= 6 && argv[2][0] != '-') r = Rect{std::atoi(argv[2]), std::atoi(argv[3]), std::atoi(argv[4]), std::atoi(argv[5])};
    const auto res = rec.recognize(gray.cview().sub(r));
    std::printf("text: \"%s\"  cost %.1f  baseline %.1f cap %.1f\n", res.utf8().c_str(), res.cost, res.baseline,
                res.cap_height);
    for (const auto& g : res.glyphs) {
        std::printf("  [%3d,%3d %2dx%2d] ", g.box.x, g.box.y, g.box.w, g.box.h);
        for (int c = 0; c < g.ncand; ++c) {
            std::string s;
            utf8_append(s, g.cand[c].ch);
            const auto& t = atlas.templates[g.cand[c].tmpl];
            std::printf("%s%s %.1f(%s %d) ", c == g.chosen ? "*" : "", s.c_str(), g.cand[c].dist,
                        atlas.fonts[t.font].c_str(), t.px);
        }
        std::printf("  eu%d la%.2f\n", g.feat.euler, g.feat.log_aspect);
        if (bench::has_flag(argc, argv, "--bmp"))
            for (int y = 0; y < 16; ++y) {
                std::printf("      ");
                for (int x = 0; x < 16; ++x) std::printf("%c", " .:-=+*#%@"[g.feat.bmp[size_t(y * 16 + x)] * 9 / 255]);
                std::printf("\n");
            }
    }
    return 0;
}
