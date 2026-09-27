// Live screen (or image) -> layout -> OCR pass over every box -> verified text.
//
//   screen_ocr                         capture the whole virtual desktop once
//   screen_ocr --frames 5              capture 5 frames (shows the memo cache warming up)
//   screen_ocr --image shot.png        run on a file instead of the screen
//   screen_ocr --out overlay.png       debug overlay (red = verified text, grey = rejected text,
//                                      blue = icon, orange = container, magenta = image)
//   screen_ocr --tsv boxes.tsv         every box: x y w h kind depth is_text confidence text
//   screen_ocr --save-frame raw.png   also save the raw captured frame
//   screen_ocr --threads 1             force single-threaded OCR
//   screen_ocr --no-verify             keep every layout Text box (no OCR verification)
#include <cstdio>
#include <fstream>

#include "common.hpp"
#include "dks/dks.hpp"
#include "dks/pipeline.hpp"

using namespace dks;

int main(int argc, char** argv) {
    SetProcessDPIAware();  // physical pixels on scaled displays
    const std::string image = bench::arg_value(argc, argv, "--image", "");
    const std::string out_png = bench::arg_value(argc, argv, "--out", "");
    const std::string out_tsv = bench::arg_value(argc, argv, "--tsv", "");
    const std::string save_frame = bench::arg_value(argc, argv, "--save-frame", "");
    const int frames = std::stoi(bench::arg_value(argc, argv, "--frames", "1"));
    const unsigned threads = unsigned(std::stoul(bench::arg_value(argc, argv, "--threads", "0")));
    const bool quiet = bench::has_flag(argc, argv, "--quiet");

    bench::Timer tl;
    ocr::Atlas atlas;
    std::string err;
    if (!atlas.load(bench::arg_value(argc, argv, "--atlas", "data/ui_fonts.dksa"), &err)) {
        std::fprintf(stderr, "atlas: %s\n", err.c_str());
        return 1;
    }
    atlas.prune(3.0f);
    const ocr::Classifier cls(atlas);
    TextVerifyParams vp;
    vp.enabled = !bench::has_flag(argc, argv, "--no-verify");
    ScreenReader reader(cls, {}, {}, vp, threads);
    reader.enable_memo(true);
    std::printf("atlas ready: %zu templates (%.0f ms), %u OCR threads\n", atlas.templates.size(), tl.ms(), reader.threads());

    for (int f = 0; f < frames; ++f) {
        bench::Timer tc;
        win32::Frame frame = image.empty() ? win32::capture_screen() : win32::load_image(image);
        if (frame.empty()) { std::fprintf(stderr, "no frame\n"); return 1; }
        const double cap_ms = tc.ms();
        if (!save_frame.empty() && f == 0) win32::save_png(save_frame, frame);

        bench::Timer tr;
        const ScreenRead SR = reader.read(frame.view());
        const double read_ms = tr.ms();
        size_t n_text = 0, n_rejected = 0, n_chars = 0;
        for (const auto& e : SR.elements) {
            if (e.is_text) ++n_text, n_chars += e.text.size();
            else if (e.element.kind == ElementKind::Text) ++n_rejected;
        }
        std::printf("frame %d: %dx%d  capture %.1f ms | layout+OCR %.1f ms | %zu boxes, %zu verified text (%zu chars), "
                    "%zu layout-text rejected by OCR\n",
                    f, frame.width, frame.height, cap_ms, read_ms, SR.elements.size(), n_text, n_chars, n_rejected);

        if (f + 1 < frames) continue;
        if (!quiet)
            for (const auto& e : SR.elements)
                if (e.is_text)
                    std::printf("  [%4d,%4d %4dx%3d] %.2f %s\n", e.element.bbox.x, e.element.bbox.y, e.element.bbox.w,
                                e.element.bbox.h, e.confidence, e.text.c_str());
        if (!out_tsv.empty()) {
            std::ofstream o(out_tsv, std::ios::binary);
            for (const auto& e : SR.elements) {
                std::string t = e.text;
                for (char& c : t)
                    if (c == '\t' || c == '\n') c = ' ';
                o << e.element.bbox.x << '\t' << e.element.bbox.y << '\t' << e.element.bbox.w << '\t' << e.element.bbox.h << '\t'
                  << kind_name(e.element.kind) << '\t' << e.element.depth << '\t' << int(e.is_text) << '\t' << e.confidence
                  << '\t' << t << '\n';
            }
        }
        if (!out_png.empty()) {
            win32::Frame o = frame;
            for (const auto& e : SR.elements) {
                const Rect& b = e.element.bbox;
                if (e.is_text) { bench::draw_rect(o, b, 255, 0, 0); continue; }
                switch (e.element.kind) {
                    case ElementKind::Text: bench::draw_rect(o, b, 150, 150, 150); break;
                    case ElementKind::Icon: bench::draw_rect(o, b, 0, 90, 255); break;
                    case ElementKind::Image: bench::draw_rect(o, b, 255, 0, 255); break;
                    case ElementKind::Container: bench::draw_rect(o, b, 255, 160, 0); break;
                }
            }
            win32::save_png(out_png, o);
        }
    }
    return 0;
}
