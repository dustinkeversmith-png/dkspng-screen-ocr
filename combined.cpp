// === C:\Users\Cutie Magic 500\projects\creative\screen-ocr\bench\bench_ocr.cpp ===
// Deterministic OCR benchmark: CER / exact match on
//   synth/seen    rendered words, atlas fonts at unseen sizes / colours
//   synth/unseen  rendered words, fonts the atlas never saw
//   icdar_bd      ICDAR 2013 Born-Digital (Challenge 1) word crops
//   webui         real browser-rendered UI text lines (WebUI sample axtree)
//
//   bench_ocr [--data datasets] [--atlas data/ui_fonts.dksa] [--vptree] [--only icdar_bd]
//             [--dump out.tsv] [--limit N] [--prune 3.0]
#include <cstdio>
#include <fstream>

#include "common.hpp"
#include "dks/dks.hpp"
#include "dks/ocr/ocr.hpp"

using namespace dks;

namespace {

struct Job {
    std::string name, dir;
    bool words_only;  // crops are single words: no space insertion
};

}  // namespace

int main(int argc, char** argv) {
    const std::string root = bench::arg_value(argc, argv, "--data", "datasets");
    const std::string atlas_path = bench::arg_value(argc, argv, "--atlas", "data/ui_fonts.dksa");
    const std::string only = bench::arg_value(argc, argv, "--only", "");
    const std::string dump = bench::arg_value(argc, argv, "--dump", "");
    const size_t limit = size_t(std::stoul(bench::arg_value(argc, argv, "--limit", "1000000")));
    const float prune = std::stof(bench::arg_value(argc, argv, "--prune", "3.0"));

    ocr::smooth_features() = !bench::has_flag(argc, argv, "--no-smooth");
    bench::Timer tl;
    ocr::Atlas atlas;
    std::string err;
    if (!atlas.load(atlas_path, &err)) { std::fprintf(stderr, "atlas: %s\n", err.c_str()); return 1; }
    const double load_ms = tl.ms();
    bench::Timer tp;
    atlas.prune(prune);
    std::printf("atlas: %zu fonts, %zu raw -> %zu templates after prune(%.1f)  [load %.0f ms, prune %.0f ms]\n",
                atlas.fonts.size(), atlas.raw_count, atlas.templates.size(), prune, load_ms, tp.ms());

    const auto mode = bench::has_flag(argc, argv, "--vptree") ? ocr::SearchMode::VPTree : ocr::SearchMode::Sorted;
    bench::Timer tb;
    ocr::Classifier cls(atlas, {}, mode);
    std::printf("classifier: %s (build %.0f ms)\n", mode == ocr::SearchMode::VPTree ? "vp-tree" : "aspect-sorted", tb.ms());
    ocr::RecognizerParams P;
    P.char_penalty = std::stof(bench::arg_value(argc, argv, "--penalty", std::to_string(P.char_penalty)));
    P.metric_weight = std::stof(bench::arg_value(argc, argv, "--metric", std::to_string(P.metric_weight)));
    P.line_metrics = !bench::has_flag(argc, argv, "--no-metrics");
    P.skip_cost = std::stof(bench::arg_value(argc, argv, "--skip", std::to_string(P.skip_cost)));
    P.space_gap = std::stof(bench::arg_value(argc, argv, "--space", std::to_string(P.space_gap)));
    P.harmonize = !bench::has_flag(argc, argv, "--no-harmonize");
    P.height_weight = std::stof(bench::arg_value(argc, argv, "--hw", std::to_string(P.height_weight)));
    P.width_norm = !bench::has_flag(argc, argv, "--no-wn");
    P.ascii_punct = bench::has_flag(argc, argv, "--ascii");
    P.char_penalty = std::stof(bench::arg_value(argc, argv, "--penalty", std::to_string(P.char_penalty)));
    P.ink.trim_margins = bench::has_flag(argc, argv, "--trim-margins");
    P.ink.sauvola = bench::has_flag(argc, argv, "--sauvola");
    P.ink.shadow_mode = !bench::has_flag(argc, argv, "--no-shadow");
    P.ink.shadow_min_dev = std::stoi(bench::arg_value(argc, argv, "--shadow-dev", "40"));
    P.ink.shadow_sharp_ratio = std::stod(bench::arg_value(argc, argv, "--shadow-ratio", "2.0"));
    P.ink.sauvola_k = std::stof(bench::arg_value(argc, argv, "--sauvola-k", "0.2"));
    P.cut_profile = std::stof(bench::arg_value(argc, argv, "--cut", std::to_string(P.cut_profile)));

    std::ofstream out;
    if (!dump.empty()) out.open(dump, std::ios::binary);

    const Job jobs[] = {{"synth/seen", root + "/synth/seen", false},
                        {"synth/unseen", root + "/synth/unseen", false},
                        {"icdar_bd", root + "/icdar_bd", true},
                        {"webui", root + "/webui", false}};
    for (const Job& job : jobs) {
        if (!only.empty() && job.name.find(only) == std::string::npos) continue;
        const auto samples = bench::load_manifest(job.dir);
        eval::TextScore score;
        double ms = 0;
        size_t glyphs = 0, skipped = 0;
        const uint64_t evals0 = cls.distance_evals, bounds0 = cls.bound_evals;
        for (const auto& s : samples) {
            if (score.n >= limit) break;
            if (job.name == "webui" && s.image.find("/sample/") == std::string::npos) continue;  // only sample has text
            win32::Frame img = win32::load_image(s.image);
            if (img.empty()) continue;
            const Gray8 gray = to_luma(img.view());
            for (const auto& g : bench::load_gt(s.gt)) {
                if (g.text.empty() || g.text == "###") continue;
                if (job.name == "webui" && g.box.h > 40) { ++skipped; continue; }  // wrapped multi-line nodes
                const Rect r = g.box.inflate(job.name == "icdar_bd" ? 0 : 2).clip(gray.width(), gray.height());
                if (r.w < 3 || r.h < 5) { ++skipped; continue; }
                ocr::RecognizerParams p = P;
                p.spaces = !job.words_only;
                const ocr::Recognizer rec(cls, p);
                bench::Timer t;
                const ocr::Recognition res = rec.recognize(gray.cview().sub(r));
                ms += t.ms();
                glyphs += res.glyphs.size();
                const std::string pred = res.utf8();
                score.add(g.text, pred);
                if (out) out << job.name << '\t' << s.image << '\t' << r.x << ',' << r.y << ',' << r.w << ',' << r.h
                             << '\t' << g.text << '\t' << pred << '\n';
            }
        }
        const double evals = double(cls.distance_evals - evals0);
        std::printf("\n== %-13s n=%zu (skipped %zu)\n", job.name.c_str(), score.n, skipped);
        std::printf("   CER %.3f  | CER(case-insens.) %.3f  | EM %.3f  | EM(ci) %.3f\n", score.cer(), score.cer_ci(),
                    score.em(), score.em_ci());
        std::printf("   %.3f ms/crop, %.1f us/glyph, %.0f distance evals/glyph\n", ms / double(score.n ? score.n : 1),
                    1000.0 * ms / double(glyphs ? glyphs : 1), evals / double(glyphs ? glyphs : 1), double(cls.bound_evals - bounds0) / double(glyphs ? glyphs : 1));
    }
    return 0;
}

// === C:\Users\Cutie Magic 500\projects\creative\screen-ocr\bench\bench_segment.cpp ===
// Element bounding + text grouping + containment benchmark on Zenodo Desktop UI and WebUI.
//
//   bench_segment [--data datasets] [--overlays out/overlays] [--edge 28] [--word 0.55] [--line 1.25]
#include <cstdio>
#include <map>
#include <memory>
#include <set>

#include "common.hpp"
#include "dks/dks.hpp"
#include "dks/pipeline.hpp"

#include <fstream>

using namespace dks;

namespace {

// Map dataset labels onto our four element kinds (or "" = ignore for per-class scores).
std::string gt_group(const std::string& dataset, const std::string& label) {
    if (label == "Text") return "Text";
    if (label == "Icon" || label == "WebIcon") return "Icon";
    if (label == "Image") return "Image";
    if (dataset == "webui") return label == "generic" ? "" : "Container";
    return "Container";  // every widget / panel label in the Zenodo taxonomy
}

const ScreenReader* g_reader = nullptr;  // OCR verification pass (default on; --no-verify)
std::ofstream g_dump;                     // --ocr-dump: per-box OCR evidence

struct DatasetScores {
    std::map<std::string, eval::DetectionScore> det;  // per group + "All"
    eval::DetectionScore text_words, text_blocks;     // GT Text vs our word / paragraph boxes
    eval::GroupingScore group_lines, group_words, group_blocks;
    eval::DepthScore depth;
    double ms = 0;
    size_t images = 0, atoms = 0;
};

// DOM-derived GT (WebUI) boxes are element boxes (full-width headings, padded buttons), not ink.
// "Tight" mode shrinks each GT box to the extent of structure pixels inside it (the same polarity-free
// edge map for every method), and drops GT boxes with no visible structure.
std::vector<bench::GtBox> tighten(const std::vector<bench::GtBox>& gt, const win32::Frame& img, int edge_t) {
    Gray8 m(img.width, img.height);
    edge_mask(img.view(), m.view(), edge_t);
    std::vector<bench::GtBox> out;
    for (auto g : gt) {
        const Rect r = g.box.clip(img.width, img.height);
        int32_t l = INT32_MAX, t = INT32_MAX, rr = INT32_MIN, b = INT32_MIN;
        for (int32_t y = r.y; y < r.bottom(); ++y)
            for (int32_t x = r.x; x < r.right(); ++x)
                if (m.at(x, y)) l = std::min(l, x), rr = std::max(rr, x + 1), t = std::min(t, y), b = std::max(b, y + 1);
        if (l == INT32_MAX || rr - l < 2 || b - t < 3) continue;
        g.box = Rect::from_ltrb(l, t, rr, b);
        out.push_back(g);
    }
    return out;
}

void run(const std::string& root, const std::string& name, const LayoutParams& P, const std::string& overlay_dir,
         int max_overlays, bool tight) {
    const auto samples = bench::load_manifest(root + "/" + name);
    if (samples.empty()) { std::printf("  (no samples for %s)\n", name.c_str()); return; }
    DatasetScores S;
    int overlays = 0;
    for (const auto& s : samples) {
        win32::Frame img = win32::load_image(s.image);
        if (img.empty()) { std::printf("  ! cannot load %s\n", s.image.c_str()); continue; }
        auto gt = bench::load_gt(s.gt);
        if (tight) gt = tighten(gt, img, P.edge_threshold);

        bench::Timer t;
        Layout L;
        std::vector<ElementKind> kind;  // effective kind after OCR verification
        std::vector<uint8_t> drop;
        if (g_reader) {
            ScreenRead SR = g_reader->read(img.view());
            L = std::move(SR.layout);
            for (const auto& re : SR.elements) {
                ElementKind k = re.element.kind;
                bool d = false;
                if (k == ElementKind::Text && !re.is_text) d = true;             // unreadable "text": drop
                if (k == ElementKind::Icon && re.is_text) k = ElementKind::Text;  // readable "icon": promote
                kind.push_back(k);
                drop.push_back(d);
                if (g_dump) {
                    float best = 0;
                    for (const auto& g : gt)
                        if (g.label == "Text") best = std::max(best, re.element.bbox.iou(g.box));
                    const auto& r = re.detail;
                    float best_any = 0;
                    for (const auto& g : gt) best_any = std::max(best_any, re.element.bbox.iou(g.box));
                    g_dump << name << '\t' << kind_name(re.element.kind) << '\t' << re.element.bbox.w << '\t'
                           << re.element.bbox.h << '\t' << best << '\t' << r.glyphs.size() << '\t' << r.mean_dist << '\t'
                           << r.alnum_frac << '\t' << r.fit_residual << '\t' << r.line_fit << '\t' << re.text << '\t' << s.image << '\t'
                           << re.element.bbox.x << '\t' << re.element.bbox.y << '\t' << best_any << '\n';
                }
            }
        } else {
            L = analyze_layout(img.view(), P);
            for (const auto& e : L.elements) kind.push_back(e.kind), drop.push_back(0);
        }
        S.ms += t.ms();
        ++S.images;
        S.atoms += L.ccl.components.size();

        std::map<std::string, std::vector<Rect>> G, Pr;
        std::vector<Rect> gt_all, pred_all, words;
        for (const auto& g : gt) {
            const std::string grp = gt_group(name, g.label);
            if (!grp.empty()) G[grp].push_back(g.box);
            gt_all.push_back(g.box);
        }
        for (size_t i = 0; i < L.elements.size(); ++i) {
            if (drop[i]) continue;
            Pr[kind_name(kind[i])].push_back(L.elements[i].bbox);
            pred_all.push_back(L.elements[i].bbox);
        }
        // Words / blocks survive only where a kept text line covers them.
        auto covered = [&](const Rect& r) {
            for (const Rect& t : Pr["Text"])
                if (t.contains_point(r.x + r.w / 2, r.y + r.h / 2)) return true;
            return false;
        };
        for (const auto& w : L.words)
            if (!g_reader || covered(w.bbox)) words.push_back(w.bbox);
        std::vector<Rect> blocks;
        for (const auto& b : L.blocks)
            if (!g_reader || covered(b.bbox)) blocks.push_back(b.bbox);
        S.text_blocks.add(G["Text"], blocks);
        S.group_blocks.add(G["Text"], blocks);

        for (const char* k : {"Text", "Icon", "Image", "Container"}) S.det[k].add(G[k], Pr[k]);
        S.det["All"].add(gt_all, pred_all);
        S.text_words.add(G["Text"], words);
        S.group_lines.add(G["Text"], Pr["Text"]);
        S.group_words.add(G["Text"], words);
        S.depth.add(gt_all, pred_all);

        if (!overlay_dir.empty() && overlays < max_overlays) {
            win32::Frame o = img;
            for (const auto& g : gt) bench::draw_rect(o, g.box, 0, 200, 0);
            for (size_t i = 0; i < L.elements.size(); ++i) {
                if (drop[i]) continue;
                const Rect& bb = L.elements[i].bbox;
                switch (kind[i]) {
                    case ElementKind::Text: bench::draw_rect(o, bb, 255, 0, 0); break;
                    case ElementKind::Icon: bench::draw_rect(o, bb, 0, 90, 255); break;
                    case ElementKind::Image: bench::draw_rect(o, bb, 255, 0, 255); break;
                    case ElementKind::Container: bench::draw_rect(o, bb, 255, 160, 0); break;
                }
            }
            char path[512];
            std::snprintf(path, sizeof path, "%s/%s_%02d.png", overlay_dir.c_str(), name.c_str(), overlays++);
            win32::save_png(path, o);
        }
    }

    std::printf("\n== %s%s%s  (%zu images, %.1f ms/frame avg, %.0f CCL atoms/frame)\n", name.c_str(), tight ? " [ink-tight GT]" : "", g_reader ? " [layout+OCR verify]" : " [layout only]", S.images,
                S.ms / double(S.images), double(S.atoms) / double(S.images));
    std::printf("  %-22s %6s %6s %8s %8s %8s %8s %8s\n", "element bounding", "n_gt", "n_pred", "R@.50", "P@.50",
                "R@.75", "F1@.50", "mIoU");
    auto row = [](const char* k, const eval::DetectionScore& d) {
        std::printf("  %-22s %6zu %6zu %8.3f %8.3f %8.3f %8.3f %8.3f\n", k, d.n_gt, d.n_pred, d.recall50(),
                    d.precision50(), d.recall75(), d.f1_50(), d.mean_best_iou());
    };
    for (const char* k : {"Text", "Icon", "Image", "Container", "All"}) row(k, S.det[k]);
    row("Text (as words)", S.text_words);
    row("Text (as blocks)", S.text_blocks);
    std::printf("  %-22s %8s %8s %8s %8s %8s\n", "text grouping", "1:1", "split", "merge", "miss", "overlap");
    auto grow = [](const char* k, const eval::GroupingScore& g) {
        std::printf("  %-22s %8.3f %8.3f %8.3f %8.3f %8.3f\n", k, g.one_to_one(), g.split_rate(), g.merge_rate(),
                    g.miss_rate(), g.overlap_ratio());
    };
    grow("lines", S.group_lines);
    grow("words", S.group_words);
    grow("blocks", S.group_blocks);
    std::printf("  containment: matched %zu, depth precision %.3f, parent precision %.3f\n", S.depth.matched,
                S.depth.depth_precision(), S.depth.parent_precision());
}

}  // namespace

int main(int argc, char** argv) {
    const std::string root = bench::arg_value(argc, argv, "--data", "datasets");
    const std::string overlays = bench::arg_value(argc, argv, "--overlays", "");
    const int max_overlays = std::stoi(bench::arg_value(argc, argv, "--max-overlays", "4"));
    LayoutParams P;
    P.edge_threshold = std::stoi(bench::arg_value(argc, argv, "--edge", std::to_string(P.edge_threshold)));
    P.word_gap = std::stof(bench::arg_value(argc, argv, "--word", std::to_string(P.word_gap)));
    P.line_gap = std::stof(bench::arg_value(argc, argv, "--line", std::to_string(P.line_gap)));
    P.image_colors = std::stoi(bench::arg_value(argc, argv, "--img-colors", std::to_string(P.image_colors)));
    P.image_min_cells = std::stoi(bench::arg_value(argc, argv, "--img-cells", std::to_string(P.image_min_cells)));
    P.flat_min_fill = std::stof(bench::arg_value(argc, argv, "--flat-fill", std::to_string(P.flat_min_fill)));
    P.icon_colors = std::stoi(bench::arg_value(argc, argv, "--icon-colors", std::to_string(P.icon_colors)));
    P.gutter_veto = !bench::has_flag(argc, argv, "--no-gutter");
    P.luma_veto = !bench::has_flag(argc, argv, "--no-luma");
    P.weak_edge_threshold = std::stoi(bench::arg_value(argc, argv, "--weak", std::to_string(P.weak_edge_threshold)));
    P.luma_delta = std::stoi(bench::arg_value(argc, argv, "--luma-delta", std::to_string(P.luma_delta)));
    P.flat_max_std = std::stof(bench::arg_value(argc, argv, "--flat-std", std::to_string(P.flat_max_std)));
    P.image_min_edge_density = std::stof(bench::arg_value(argc, argv, "--img-edges", std::to_string(P.image_min_edge_density)));
    P.image_min_size = std::stoi(bench::arg_value(argc, argv, "--img-min", std::to_string(P.image_min_size)));
    P.image_min_fill = std::stof(bench::arg_value(argc, argv, "--img-fill", std::to_string(P.image_min_fill)));
    P.detect_images = !bench::has_flag(argc, argv, "--no-images");
    P.flat_containers = !bench::has_flag(argc, argv, "--no-flat");
    std::unique_ptr<ocr::Atlas> atlas;
    std::unique_ptr<ocr::Classifier> cls;
    std::unique_ptr<ScreenReader> reader;
    if (!bench::has_flag(argc, argv, "--no-verify")) {
        atlas = std::make_unique<ocr::Atlas>();
        std::string err;
        if (!atlas->load(bench::arg_value(argc, argv, "--atlas", "data/ui_fonts.dksa"), &err)) {
            std::fprintf(stderr, "atlas: %s (use --no-verify to skip OCR)\n", err.c_str());
            return 1;
        }
        atlas->prune(3.0f);
        cls = std::make_unique<ocr::Classifier>(*atlas);
        cls->enable_memo(true);
        reader = std::make_unique<ScreenReader>(*cls, P);
        g_reader = reader.get();
        const std::string dump = bench::arg_value(argc, argv, "--ocr-dump", "");
        if (!dump.empty()) g_dump.open(dump, std::ios::binary);
    }
    if (!overlays.empty()) CreateDirectoryA(overlays.c_str(), nullptr);
    const std::string only = bench::arg_value(argc, argv, "--only", "");
    if (!only.empty() && only != "zenodo" && only != "webui") {  // any other manifest dir, e.g. negatives
        run(root, only, P, overlays, max_overlays, false);
        return 0;
    }
    for (const char* ds : {"zenodo", "webui"})
        if (only.empty() || only == ds) {
            run(root, ds, P, overlays, max_overlays, false);
            if (std::string(ds) == "webui") run(root, ds, P, "", 0, true);
        }
    return 0;
}

// === C:\Users\Cutie Magic 500\projects\creative\screen-ocr\bench\ocr_debug.cpp ===
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

// === C:\Users\Cutie Magic 500\projects\creative\screen-ocr\bench\screen_ocr.cpp ===
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

// === C:\Users\Cutie Magic 500\projects\creative\screen-ocr\build\CMakeFiles\4.3.3\CompilerIdCXX\CMakeCXXCompilerId.cpp ===
/* This source file must have a .cpp extension so that all C++ compilers
   recognize the extension without flags.  Borland does not know .cxx for
   example.  */
#ifndef __cplusplus
# error "A C compiler has been selected for C++."
#endif

#if !defined(__has_include)
/* If the compiler does not have __has_include, pretend the answer is
   always no.  */
#  define __has_include(x) 0
#endif


/* Version number components: V=Version, R=Revision, P=Patch
   Version date components:   YYYY=Year, MM=Month,   DD=Day  */

#if defined(__INTEL_COMPILER) || defined(__ICC)
# define COMPILER_ID "Intel"
# if defined(_MSC_VER)
#  define SIMULATE_ID "MSVC"
# endif
# if defined(__GNUC__)
#  define SIMULATE_ID "GNU"
# endif
  /* __INTEL_COMPILER = VRP prior to 2021, and then VVVV for 2021 and later,
     except that a few beta releases use the old format with V=2021.  */
# if __INTEL_COMPILER < 2021 || __INTEL_COMPILER == 202110 || __INTEL_COMPILER == 202111
#  define COMPILER_VERSION_MAJOR DEC(__INTEL_COMPILER/100)
#  define COMPILER_VERSION_MINOR DEC(__INTEL_COMPILER/10 % 10)
#  if defined(__INTEL_COMPILER_UPDATE)
#   define COMPILER_VERSION_PATCH DEC(__INTEL_COMPILER_UPDATE)
#  else
#   define COMPILER_VERSION_PATCH DEC(__INTEL_COMPILER   % 10)
#  endif
# else
#  define COMPILER_VERSION_MAJOR DEC(__INTEL_COMPILER)
#  define COMPILER_VERSION_MINOR DEC(__INTEL_COMPILER_UPDATE)
   /* The third version component from --version is an update index,
      but no macro is provided for it.  */
#  define COMPILER_VERSION_PATCH DEC(0)
# endif
# if defined(__INTEL_COMPILER_BUILD_DATE)
   /* __INTEL_COMPILER_BUILD_DATE = YYYYMMDD */
#  define COMPILER_VERSION_TWEAK DEC(__INTEL_COMPILER_BUILD_DATE)
# endif
# if defined(_MSC_VER)
   /* _MSC_VER = VVRR */
#  define SIMULATE_VERSION_MAJOR DEC(_MSC_VER / 100)
#  define SIMULATE_VERSION_MINOR DEC(_MSC_VER % 100)
# endif
# if defined(__GNUC__)
#  define SIMULATE_VERSION_MAJOR DEC(__GNUC__)
# elif defined(__GNUG__)
#  define SIMULATE_VERSION_MAJOR DEC(__GNUG__)
# endif
# if defined(__GNUC_MINOR__)
#  define SIMULATE_VERSION_MINOR DEC(__GNUC_MINOR__)
# endif
# if defined(__GNUC_PATCHLEVEL__)
#  define SIMULATE_VERSION_PATCH DEC(__GNUC_PATCHLEVEL__)
# endif

#elif (defined(__clang__) && defined(__INTEL_CLANG_COMPILER)) || defined(__INTEL_LLVM_COMPILER)
# define COMPILER_ID "IntelLLVM"
#if defined(_MSC_VER)
# define SIMULATE_ID "MSVC"
#endif
#if defined(__GNUC__)
# define SIMULATE_ID "GNU"
#endif
/* __INTEL_LLVM_COMPILER = VVVVRP prior to 2021.2.0, VVVVRRPP for 2021.2.0 and
 * later.  Look for 6 digit vs. 8 digit version number to decide encoding.
 * VVVV is no smaller than the current year when a version is released.
 */
#if __INTEL_LLVM_COMPILER < 1000000L
# define COMPILER_VERSION_MAJOR DEC(__INTEL_LLVM_COMPILER/100)
# define COMPILER_VERSION_MINOR DEC(__INTEL_LLVM_COMPILER/10 % 10)
# define COMPILER_VERSION_PATCH DEC(__INTEL_LLVM_COMPILER    % 10)
#else
# define COMPILER_VERSION_MAJOR DEC(__INTEL_LLVM_COMPILER/10000)
# define COMPILER_VERSION_MINOR DEC(__INTEL_LLVM_COMPILER/100 % 100)
# define COMPILER_VERSION_PATCH DEC(__INTEL_LLVM_COMPILER     % 100)
#endif
#if defined(_MSC_VER)
  /* _MSC_VER = VVRR */
# define SIMULATE_VERSION_MAJOR DEC(_MSC_VER / 100)
# define SIMULATE_VERSION_MINOR DEC(_MSC_VER % 100)
#endif
#if defined(__GNUC__)
# define SIMULATE_VERSION_MAJOR DEC(__GNUC__)
#elif defined(__GNUG__)
# define SIMULATE_VERSION_MAJOR DEC(__GNUG__)
#endif
#if defined(__GNUC_MINOR__)
# define SIMULATE_VERSION_MINOR DEC(__GNUC_MINOR__)
#endif
#if defined(__GNUC_PATCHLEVEL__)
# define SIMULATE_VERSION_PATCH DEC(__GNUC_PATCHLEVEL__)
#endif

#elif defined(__PATHCC__)
# define COMPILER_ID "PathScale"
# define COMPILER_VERSION_MAJOR DEC(__PATHCC__)
# define COMPILER_VERSION_MINOR DEC(__PATHCC_MINOR__)
# if defined(__PATHCC_PATCHLEVEL__)
#  define COMPILER_VERSION_PATCH DEC(__PATHCC_PATCHLEVEL__)
# endif

#elif defined(__BORLANDC__) && defined(__CODEGEARC_VERSION__)
# define COMPILER_ID "Embarcadero"
# define COMPILER_VERSION_MAJOR HEX(__CODEGEARC_VERSION__>>24 & 0x00FF)
# define COMPILER_VERSION_MINOR HEX(__CODEGEARC_VERSION__>>16 & 0x00FF)
# define COMPILER_VERSION_PATCH DEC(__CODEGEARC_VERSION__     & 0xFFFF)

#elif defined(__BORLANDC__)
# define COMPILER_ID "Borland"
  /* __BORLANDC__ = 0xVRR */
# define COMPILER_VERSION_MAJOR HEX(__BORLANDC__>>8)
# define COMPILER_VERSION_MINOR HEX(__BORLANDC__ & 0xFF)

#elif defined(__WATCOMC__) && __WATCOMC__ < 1200
# define COMPILER_ID "Watcom"
   /* __WATCOMC__ = VVRR */
# define COMPILER_VERSION_MAJOR DEC(__WATCOMC__ / 100)
# define COMPILER_VERSION_MINOR DEC((__WATCOMC__ / 10) % 10)
# if (__WATCOMC__ % 10) > 0
#  define COMPILER_VERSION_PATCH DEC(__WATCOMC__ % 10)
# endif

#elif defined(__WATCOMC__)
# define COMPILER_ID "OpenWatcom"
   /* __WATCOMC__ = VVRP + 1100 */
# define COMPILER_VERSION_MAJOR DEC((__WATCOMC__ - 1100) / 100)
# define COMPILER_VERSION_MINOR DEC((__WATCOMC__ / 10) % 10)
# if (__WATCOMC__ % 10) > 0
#  define COMPILER_VERSION_PATCH DEC(__WATCOMC__ % 10)
# endif

#elif defined(__SUNPRO_CC)
# define COMPILER_ID "SunPro"
# if __SUNPRO_CC >= 0x5100
   /* __SUNPRO_CC = 0xVRRP */
#  define COMPILER_VERSION_MAJOR HEX(__SUNPRO_CC>>12)
#  define COMPILER_VERSION_MINOR HEX(__SUNPRO_CC>>4 & 0xFF)
#  define COMPILER_VERSION_PATCH HEX(__SUNPRO_CC    & 0xF)
# else
   /* __SUNPRO_CC = 0xVRP */
#  define COMPILER_VERSION_MAJOR HEX(__SUNPRO_CC>>8)
#  define COMPILER_VERSION_MINOR HEX(__SUNPRO_CC>>4 & 0xF)
#  define COMPILER_VERSION_PATCH HEX(__SUNPRO_CC    & 0xF)
# endif

#elif defined(__HP_aCC)
# define COMPILER_ID "HP"
  /* __HP_aCC = VVRRPP */
# define COMPILER_VERSION_MAJOR DEC(__HP_aCC/10000)
# define COMPILER_VERSION_MINOR DEC(__HP_aCC/100 % 100)
# define COMPILER_VERSION_PATCH DEC(__HP_aCC     % 100)

#elif defined(__DECCXX)
# define COMPILER_ID "Compaq"
  /* __DECCXX_VER = VVRRTPPPP */
# define COMPILER_VERSION_MAJOR DEC(__DECCXX_VER/10000000)
# define COMPILER_VERSION_MINOR DEC(__DECCXX_VER/100000  % 100)
# define COMPILER_VERSION_PATCH DEC(__DECCXX_VER         % 10000)

#elif defined(__IBMCPP__) && defined(__COMPILER_VER__)
# define COMPILER_ID "zOS"
  /* __IBMCPP__ = VRP */
# define COMPILER_VERSION_MAJOR DEC(__IBMCPP__/100)
# define COMPILER_VERSION_MINOR DEC(__IBMCPP__/10 % 10)
# define COMPILER_VERSION_PATCH DEC(__IBMCPP__    % 10)

#elif defined(__open_xl__) && defined(__clang__)
# define COMPILER_ID "IBMClang"
# define COMPILER_VERSION_MAJOR DEC(__open_xl_version__)
# define COMPILER_VERSION_MINOR DEC(__open_xl_release__)
# define COMPILER_VERSION_PATCH DEC(__open_xl_modification__)
# define COMPILER_VERSION_TWEAK DEC(__open_xl_ptf_fix_level__)
# define COMPILER_VERSION_INTERNAL_STR  __clang_version__


#elif defined(__ibmxl__) && defined(__clang__)
# define COMPILER_ID "XLClang"
# define COMPILER_VERSION_MAJOR DEC(__ibmxl_version__)
# define COMPILER_VERSION_MINOR DEC(__ibmxl_release__)
# define COMPILER_VERSION_PATCH DEC(__ibmxl_modification__)
# define COMPILER_VERSION_TWEAK DEC(__ibmxl_ptf_fix_level__)


#elif defined(__IBMCPP__) && !defined(__COMPILER_VER__) && __IBMCPP__ >= 800
# define COMPILER_ID "XL"
  /* __IBMCPP__ = VRP */
# define COMPILER_VERSION_MAJOR DEC(__IBMCPP__/100)
# define COMPILER_VERSION_MINOR DEC(__IBMCPP__/10 % 10)
# define COMPILER_VERSION_PATCH DEC(__IBMCPP__    % 10)

#elif defined(__IBMCPP__) && !defined(__COMPILER_VER__) && __IBMCPP__ < 800
# define COMPILER_ID "VisualAge"
  /* __IBMCPP__ = VRP */
# define COMPILER_VERSION_MAJOR DEC(__IBMCPP__/100)
# define COMPILER_VERSION_MINOR DEC(__IBMCPP__/10 % 10)
# define COMPILER_VERSION_PATCH DEC(__IBMCPP__    % 10)

#elif defined(__NVCOMPILER)
# define COMPILER_ID "NVHPC"
# define COMPILER_VERSION_MAJOR DEC(__NVCOMPILER_MAJOR__)
# define COMPILER_VERSION_MINOR DEC(__NVCOMPILER_MINOR__)
# if defined(__NVCOMPILER_PATCHLEVEL__)
#  define COMPILER_VERSION_PATCH DEC(__NVCOMPILER_PATCHLEVEL__)
# endif

#elif defined(__PGI)
# define COMPILER_ID "PGI"
# define COMPILER_VERSION_MAJOR DEC(__PGIC__)
# define COMPILER_VERSION_MINOR DEC(__PGIC_MINOR__)
# if defined(__PGIC_PATCHLEVEL__)
#  define COMPILER_VERSION_PATCH DEC(__PGIC_PATCHLEVEL__)
# endif

#elif defined(__clang__) && defined(__cray__)
# define COMPILER_ID "CrayClang"
# define COMPILER_VERSION_MAJOR DEC(__cray_major__)
# define COMPILER_VERSION_MINOR DEC(__cray_minor__)
# define COMPILER_VERSION_PATCH DEC(__cray_patchlevel__)
# define COMPILER_VERSION_INTERNAL_STR __clang_version__


#elif defined(_CRAYC)
# define COMPILER_ID "Cray"
# define COMPILER_VERSION_MAJOR DEC(_RELEASE_MAJOR)
# define COMPILER_VERSION_MINOR DEC(_RELEASE_MINOR)

#elif defined(__TI_COMPILER_VERSION__)
# define COMPILER_ID "TI"
  /* __TI_COMPILER_VERSION__ = VVVRRRPPP */
# define COMPILER_VERSION_MAJOR DEC(__TI_COMPILER_VERSION__/1000000)
# define COMPILER_VERSION_MINOR DEC(__TI_COMPILER_VERSION__/1000   % 1000)
# define COMPILER_VERSION_PATCH DEC(__TI_COMPILER_VERSION__        % 1000)

#elif defined(__CLANG_FUJITSU)
# define COMPILER_ID "FujitsuClang"
# define COMPILER_VERSION_MAJOR DEC(__FCC_major__)
# define COMPILER_VERSION_MINOR DEC(__FCC_minor__)
# define COMPILER_VERSION_PATCH DEC(__FCC_patchlevel__)
# define COMPILER_VERSION_INTERNAL_STR __clang_version__


#elif defined(__FUJITSU)
# define COMPILER_ID "Fujitsu"
# if defined(__FCC_version__)
#   define COMPILER_VERSION __FCC_version__
# elif defined(__FCC_major__)
#   define COMPILER_VERSION_MAJOR DEC(__FCC_major__)
#   define COMPILER_VERSION_MINOR DEC(__FCC_minor__)
#   define COMPILER_VERSION_PATCH DEC(__FCC_patchlevel__)
# endif
# if defined(__fcc_version)
#   define COMPILER_VERSION_INTERNAL DEC(__fcc_version)
# elif defined(__FCC_VERSION)
#   define COMPILER_VERSION_INTERNAL DEC(__FCC_VERSION)
# endif


#elif defined(__ghs__)
# define COMPILER_ID "GHS"
/* __GHS_VERSION_NUMBER = VVVVRP */
# ifdef __GHS_VERSION_NUMBER
# define COMPILER_VERSION_MAJOR DEC(__GHS_VERSION_NUMBER / 100)
# define COMPILER_VERSION_MINOR DEC(__GHS_VERSION_NUMBER / 10 % 10)
# define COMPILER_VERSION_PATCH DEC(__GHS_VERSION_NUMBER      % 10)
# endif

#elif defined(__TASKING__)
# define COMPILER_ID "Tasking"
  # define COMPILER_VERSION_MAJOR DEC(__VERSION__/1000)
  # define COMPILER_VERSION_MINOR DEC(__VERSION__ % 100)
# define COMPILER_VERSION_INTERNAL DEC(__VERSION__)

#elif defined(__ORANGEC__)
# define COMPILER_ID "OrangeC"
# define COMPILER_VERSION_MAJOR DEC(__ORANGEC_MAJOR__)
# define COMPILER_VERSION_MINOR DEC(__ORANGEC_MINOR__)
# define COMPILER_VERSION_PATCH DEC(__ORANGEC_PATCHLEVEL__)

#elif defined(__RENESAS__)
# define COMPILER_ID "Renesas"
/* __RENESAS_VERSION__ = 0xVVRRPP00 */
# define COMPILER_VERSION_MAJOR HEX(__RENESAS_VERSION__ >> 24 & 0xFF)
# define COMPILER_VERSION_MINOR HEX(__RENESAS_VERSION__ >> 16 & 0xFF)
# define COMPILER_VERSION_PATCH HEX(__RENESAS_VERSION__ >> 8  & 0xFF)

#elif defined(__SCO_VERSION__)
# define COMPILER_ID "SCO"

#elif defined(__ARMCC_VERSION) && !defined(__clang__)
# define COMPILER_ID "ARMCC"
#if __ARMCC_VERSION >= 1000000
  /* __ARMCC_VERSION = VRRPPPP */
  # define COMPILER_VERSION_MAJOR DEC(__ARMCC_VERSION/1000000)
  # define COMPILER_VERSION_MINOR DEC(__ARMCC_VERSION/10000 % 100)
  # define COMPILER_VERSION_PATCH DEC(__ARMCC_VERSION     % 10000)
#else
  /* __ARMCC_VERSION = VRPPPP */
  # define COMPILER_VERSION_MAJOR DEC(__ARMCC_VERSION/100000)
  # define COMPILER_VERSION_MINOR DEC(__ARMCC_VERSION/10000 % 10)
  # define COMPILER_VERSION_PATCH DEC(__ARMCC_VERSION    % 10000)
#endif


#elif defined(__clang__) && defined(__apple_build_version__)
# define COMPILER_ID "AppleClang"
# if defined(_MSC_VER)
#  define SIMULATE_ID "MSVC"
# endif
# define COMPILER_VERSION_MAJOR DEC(__clang_major__)
# define COMPILER_VERSION_MINOR DEC(__clang_minor__)
# define COMPILER_VERSION_PATCH DEC(__clang_patchlevel__)
# if defined(_MSC_VER)
   /* _MSC_VER = VVRR */
#  define SIMULATE_VERSION_MAJOR DEC(_MSC_VER / 100)
#  define SIMULATE_VERSION_MINOR DEC(_MSC_VER % 100)
# endif
# define COMPILER_VERSION_TWEAK DEC(__apple_build_version__)

#elif defined(__clang__) && defined(__ARMCOMPILER_VERSION)
# define COMPILER_ID "ARMClang"
  # define COMPILER_VERSION_MAJOR DEC(__ARMCOMPILER_VERSION/1000000)
  # define COMPILER_VERSION_MINOR DEC(__ARMCOMPILER_VERSION/10000 % 100)
  # define COMPILER_VERSION_PATCH DEC(__ARMCOMPILER_VERSION/100   % 100)
# define COMPILER_VERSION_INTERNAL DEC(__ARMCOMPILER_VERSION)

#elif defined(__clang__) && defined(__ti__)
# define COMPILER_ID "TIClang"
  # define COMPILER_VERSION_MAJOR DEC(__ti_major__)
  # define COMPILER_VERSION_MINOR DEC(__ti_minor__)
  # define COMPILER_VERSION_PATCH DEC(__ti_patchlevel__)
# define COMPILER_VERSION_INTERNAL DEC(__ti_version__)

#elif defined(__clang__)
# define COMPILER_ID "Clang"
# if defined(_MSC_VER)
#  define SIMULATE_ID "MSVC"
# endif
# define COMPILER_VERSION_MAJOR DEC(__clang_major__)
# define COMPILER_VERSION_MINOR DEC(__clang_minor__)
# define COMPILER_VERSION_PATCH DEC(__clang_patchlevel__)
# if defined(_MSC_VER)
   /* _MSC_VER = VVRR */
#  define SIMULATE_VERSION_MAJOR DEC(_MSC_VER / 100)
#  define SIMULATE_VERSION_MINOR DEC(_MSC_VER % 100)
# endif

#elif defined(__LCC__) && (defined(__GNUC__) || defined(__GNUG__) || defined(__MCST__))
# define COMPILER_ID "LCC"
# define COMPILER_VERSION_MAJOR DEC(__LCC__ / 100)
# define COMPILER_VERSION_MINOR DEC(__LCC__ % 100)
# if defined(__LCC_MINOR__)
#  define COMPILER_VERSION_PATCH DEC(__LCC_MINOR__)
# endif
# if defined(__GNUC__) && defined(__GNUC_MINOR__)
#  define SIMULATE_ID "GNU"
#  define SIMULATE_VERSION_MAJOR DEC(__GNUC__)
#  define SIMULATE_VERSION_MINOR DEC(__GNUC_MINOR__)
#  if defined(__GNUC_PATCHLEVEL__)
#   define SIMULATE_VERSION_PATCH DEC(__GNUC_PATCHLEVEL__)
#  endif
# endif

#elif defined(__GNUC__) || defined(__GNUG__)
# define COMPILER_ID "GNU"
# if defined(__GNUC__)
#  define COMPILER_VERSION_MAJOR DEC(__GNUC__)
# else
#  define COMPILER_VERSION_MAJOR DEC(__GNUG__)
# endif
# if defined(__GNUC_MINOR__)
#  define COMPILER_VERSION_MINOR DEC(__GNUC_MINOR__)
# endif
# if defined(__GNUC_PATCHLEVEL__)
#  define COMPILER_VERSION_PATCH DEC(__GNUC_PATCHLEVEL__)
# endif

#elif defined(_MSC_VER)
# define COMPILER_ID "MSVC"
  /* _MSC_VER = VVRR */
# define COMPILER_VERSION_MAJOR DEC(_MSC_VER / 100)
# define COMPILER_VERSION_MINOR DEC(_MSC_VER % 100)
# if defined(_MSC_FULL_VER)
#  if _MSC_VER >= 1400
    /* _MSC_FULL_VER = VVRRPPPPP */
#   define COMPILER_VERSION_PATCH DEC(_MSC_FULL_VER % 100000)
#  else
    /* _MSC_FULL_VER = VVRRPPPP */
#   define COMPILER_VERSION_PATCH DEC(_MSC_FULL_VER % 10000)
#  endif
# endif
# if defined(_MSC_BUILD)
#  define COMPILER_VERSION_TWEAK DEC(_MSC_BUILD)
# endif

#elif defined(_ADI_COMPILER)
# define COMPILER_ID "ADSP"
#if defined(__VERSIONNUM__)
  /* __VERSIONNUM__ = 0xVVRRPPTT */
#  define COMPILER_VERSION_MAJOR DEC(__VERSIONNUM__ >> 24 & 0xFF)
#  define COMPILER_VERSION_MINOR DEC(__VERSIONNUM__ >> 16 & 0xFF)
#  define COMPILER_VERSION_PATCH DEC(__VERSIONNUM__ >> 8 & 0xFF)
#  define COMPILER_VERSION_TWEAK DEC(__VERSIONNUM__ & 0xFF)
#endif

#elif defined(__IAR_SYSTEMS_ICC__) || defined(__IAR_SYSTEMS_ICC)
# define COMPILER_ID "IAR"
# if defined(__VER__) && defined(__ICCARM__)
#  define COMPILER_VERSION_MAJOR DEC((__VER__) / 1000000)
#  define COMPILER_VERSION_MINOR DEC(((__VER__) / 1000) % 1000)
#  define COMPILER_VERSION_PATCH DEC((__VER__) % 1000)
#  define COMPILER_VERSION_INTERNAL DEC(__IAR_SYSTEMS_ICC__)
# elif defined(__VER__) && (defined(__ICCAVR__) || defined(__ICCRX__) || defined(__ICCRH850__) || defined(__ICCRL78__) || defined(__ICC430__) || defined(__ICCRISCV__) || defined(__ICCV850__) || defined(__ICC8051__) || defined(__ICCSTM8__))
#  define COMPILER_VERSION_MAJOR DEC((__VER__) / 100)
#  define COMPILER_VERSION_MINOR DEC((__VER__) - (((__VER__) / 100)*100))
#  define COMPILER_VERSION_PATCH DEC(__SUBVERSION__)
#  define COMPILER_VERSION_INTERNAL DEC(__IAR_SYSTEMS_ICC__)
# endif

#elif defined(__DCC__) && defined(_DIAB_TOOL)
# define COMPILER_ID "Diab"
  # define COMPILER_VERSION_MAJOR DEC(__VERSION_MAJOR_NUMBER__)
  # define COMPILER_VERSION_MINOR DEC(__VERSION_MINOR_NUMBER__)
  # define COMPILER_VERSION_PATCH DEC(__VERSION_ARCH_FEATURE_NUMBER__)
  # define COMPILER_VERSION_TWEAK DEC(__VERSION_BUG_FIX_NUMBER__)



/* These compilers are either not known or too old to define an
  identification macro.  Try to identify the platform and guess that
  it is the native compiler.  */
#elif defined(__hpux) || defined(__hpua)
# define COMPILER_ID "HP"

#else /* unknown compiler */
# define COMPILER_ID ""
#endif

/* Construct the string literal in pieces to prevent the source from
   getting matched.  Store it in a pointer rather than an array
   because some compilers will just produce instructions to fill the
   array rather than assigning a pointer to a static array.  */
char const* info_compiler = "INFO" ":" "compiler[" COMPILER_ID "]";
#ifdef SIMULATE_ID
char const* info_simulate = "INFO" ":" "simulate[" SIMULATE_ID "]";
#endif

#ifdef __QNXNTO__
char const* qnxnto = "INFO" ":" "qnxnto[]";
#endif

#if defined(__CRAYXT_COMPUTE_LINUX_TARGET)
char const *info_cray = "INFO" ":" "compiler_wrapper[CrayPrgEnv]";
#endif

#define STRINGIFY_HELPER(X) #X
#define STRINGIFY(X) STRINGIFY_HELPER(X)

/* Identify known platforms by name.  */
#if defined(__linux) || defined(__linux__) || defined(linux)
# define PLATFORM_ID "Linux"

#elif defined(__MSYS__)
# define PLATFORM_ID "MSYS"

#elif defined(__CYGWIN__)
# define PLATFORM_ID "Cygwin"

#elif defined(__MINGW32__)
# define PLATFORM_ID "MinGW"

#elif defined(__APPLE__)
# define PLATFORM_ID "Darwin"

#elif defined(_WIN32) || defined(__WIN32__) || defined(WIN32)
# define PLATFORM_ID "Windows"

#elif defined(__FreeBSD__) || defined(__FreeBSD)
# define PLATFORM_ID "FreeBSD"

#elif defined(__NetBSD__) || defined(__NetBSD)
# define PLATFORM_ID "NetBSD"

#elif defined(__OpenBSD__) || defined(__OPENBSD)
# define PLATFORM_ID "OpenBSD"

#elif defined(__sun) || defined(sun)
# define PLATFORM_ID "SunOS"

#elif defined(_AIX) || defined(__AIX) || defined(__AIX__) || defined(__aix) || defined(__aix__)
# define PLATFORM_ID "AIX"

#elif defined(__hpux) || defined(__hpux__)
# define PLATFORM_ID "HP-UX"

#elif defined(__HAIKU__)
# define PLATFORM_ID "Haiku"

#elif defined(__BeOS) || defined(__BEOS__) || defined(_BEOS)
# define PLATFORM_ID "BeOS"

#elif defined(__QNX__) || defined(__QNXNTO__)
# define PLATFORM_ID "QNX"

#elif defined(__tru64) || defined(_tru64) || defined(__TRU64__)
# define PLATFORM_ID "Tru64"

#elif defined(__riscos) || defined(__riscos__)
# define PLATFORM_ID "RISCos"

#elif defined(__sinix) || defined(__sinix__) || defined(__SINIX__)
# define PLATFORM_ID "SINIX"

#elif defined(__UNIX_SV__)
# define PLATFORM_ID "UNIX_SV"

#elif defined(__bsdos__)
# define PLATFORM_ID "BSDOS"

#elif defined(_MPRAS) || defined(MPRAS)
# define PLATFORM_ID "MP-RAS"

#elif defined(__osf) || defined(__osf__)
# define PLATFORM_ID "OSF1"

#elif defined(_SCO_SV) || defined(SCO_SV) || defined(sco_sv)
# define PLATFORM_ID "SCO_SV"

#elif defined(__ultrix) || defined(__ultrix__) || defined(_ULTRIX)
# define PLATFORM_ID "ULTRIX"

#elif defined(__XENIX__) || defined(_XENIX) || defined(XENIX)
# define PLATFORM_ID "Xenix"

#elif defined(__WATCOMC__)
# if defined(__LINUX__)
#  define PLATFORM_ID "Linux"

# elif defined(__DOS__)
#  define PLATFORM_ID "DOS"

# elif defined(__OS2__)
#  define PLATFORM_ID "OS2"

# elif defined(__WINDOWS__)
#  define PLATFORM_ID "Windows3x"

# elif defined(__VXWORKS__)
#  define PLATFORM_ID "VxWorks"

# else /* unknown platform */
#  define PLATFORM_ID
# endif

#elif defined(__INTEGRITY)
# if defined(INT_178B)
#  define PLATFORM_ID "Integrity178"

# else /* regular Integrity */
#  define PLATFORM_ID "Integrity"
# endif

# elif defined(_ADI_COMPILER)
#  define PLATFORM_ID "ADSP"

#else /* unknown platform */
# define PLATFORM_ID

#endif

/* For windows compilers MSVC and Intel we can determine
   the architecture of the compiler being used.  This is because
   the compilers do not have flags that can change the architecture,
   but rather depend on which compiler is being used
*/
#if defined(_WIN32) && defined(_MSC_VER)
# if defined(_M_IA64)
#  define ARCHITECTURE_ID "IA64"

# elif defined(_M_ARM64EC)
#  define ARCHITECTURE_ID "ARM64EC"

# elif defined(_M_X64) || defined(_M_AMD64)
#  define ARCHITECTURE_ID "x64"

# elif defined(_M_IX86)
#  define ARCHITECTURE_ID "X86"

# elif defined(_M_ARM64)
#  define ARCHITECTURE_ID "ARM64"

# elif defined(_M_ARM)
#  if _M_ARM == 4
#   define ARCHITECTURE_ID "ARMV4I"
#  elif _M_ARM == 5
#   define ARCHITECTURE_ID "ARMV5I"
#  else
#   define ARCHITECTURE_ID "ARMV" STRINGIFY(_M_ARM)
#  endif

# elif defined(_M_MIPS)
#  define ARCHITECTURE_ID "MIPS"

# elif defined(_M_SH)
#  define ARCHITECTURE_ID "SHx"

# else /* unknown architecture */
#  define ARCHITECTURE_ID ""
# endif

#elif defined(__WATCOMC__)
# if defined(_M_I86)
#  define ARCHITECTURE_ID "I86"

# elif defined(_M_IX86)
#  define ARCHITECTURE_ID "X86"

# else /* unknown architecture */
#  define ARCHITECTURE_ID ""
# endif

#elif defined(__IAR_SYSTEMS_ICC__) || defined(__IAR_SYSTEMS_ICC)
# if defined(__ICCARM__)
#  define ARCHITECTURE_ID "ARM"

# elif defined(__ICCRX__)
#  define ARCHITECTURE_ID "RX"

# elif defined(__ICCRH850__)
#  define ARCHITECTURE_ID "RH850"

# elif defined(__ICCRL78__)
#  define ARCHITECTURE_ID "RL78"

# elif defined(__ICCRISCV__)
#  define ARCHITECTURE_ID "RISCV"

# elif defined(__ICCAVR__)
#  define ARCHITECTURE_ID "AVR"

# elif defined(__ICC430__)
#  define ARCHITECTURE_ID "MSP430"

# elif defined(__ICCV850__)
#  define ARCHITECTURE_ID "V850"

# elif defined(__ICC8051__)
#  define ARCHITECTURE_ID "8051"

# elif defined(__ICCSTM8__)
#  define ARCHITECTURE_ID "STM8"

# else /* unknown architecture */
#  define ARCHITECTURE_ID ""
# endif

#elif defined(__ghs__)
# if defined(__PPC64__)
#  define ARCHITECTURE_ID "PPC64"

# elif defined(__ppc__)
#  define ARCHITECTURE_ID "PPC"

# elif defined(__ARM__)
#  define ARCHITECTURE_ID "ARM"

# elif defined(__x86_64__)
#  define ARCHITECTURE_ID "x64"

# elif defined(__i386__)
#  define ARCHITECTURE_ID "X86"

# else /* unknown architecture */
#  define ARCHITECTURE_ID ""
# endif

#elif defined(__clang__) && defined(__ti__)
# if defined(__ARM_ARCH)
#  define ARCHITECTURE_ID "ARM"

# else /* unknown architecture */
#  define ARCHITECTURE_ID ""
# endif

#elif defined(__TI_COMPILER_VERSION__)
# if defined(__TI_ARM__)
#  define ARCHITECTURE_ID "ARM"

# elif defined(__MSP430__)
#  define ARCHITECTURE_ID "MSP430"

# elif defined(__TMS320C28XX__)
#  define ARCHITECTURE_ID "TMS320C28x"

# elif defined(__TMS320C6X__) || defined(_TMS320C6X)
#  define ARCHITECTURE_ID "TMS320C6x"

# else /* unknown architecture */
#  define ARCHITECTURE_ID ""
# endif

# elif defined(__ADSPSHARC__)
#  define ARCHITECTURE_ID "SHARC"

# elif defined(__ADSPBLACKFIN__)
#  define ARCHITECTURE_ID "Blackfin"

#elif defined(__TASKING__)

# if defined(__CTC__) || defined(__CPTC__)
#  define ARCHITECTURE_ID "TriCore"

# elif defined(__CMCS__)
#  define ARCHITECTURE_ID "MCS"

# elif defined(__CARM__) || defined(__CPARM__)
#  define ARCHITECTURE_ID "ARM"

# elif defined(__CARC__)
#  define ARCHITECTURE_ID "ARC"

# elif defined(__C51__)
#  define ARCHITECTURE_ID "8051"

# elif defined(__CPCP__)
#  define ARCHITECTURE_ID "PCP"

# else
#  define ARCHITECTURE_ID ""
# endif

#elif defined(__RENESAS__)
# if defined(__CCRX__)
#  define ARCHITECTURE_ID "RX"

# elif defined(__CCRL__)
#  define ARCHITECTURE_ID "RL78"

# elif defined(__CCRH__)
#  define ARCHITECTURE_ID "RH850"

# else
#  define ARCHITECTURE_ID ""
# endif

#else
#  define ARCHITECTURE_ID
#endif

/* Convert integer to decimal digit literals.  */
#define DEC(n)                   \
  ('0' + (((n) / 10000000)%10)), \
  ('0' + (((n) / 1000000)%10)),  \
  ('0' + (((n) / 100000)%10)),   \
  ('0' + (((n) / 10000)%10)),    \
  ('0' + (((n) / 1000)%10)),     \
  ('0' + (((n) / 100)%10)),      \
  ('0' + (((n) / 10)%10)),       \
  ('0' +  ((n) % 10))

/* Convert integer to hex digit literals.  */
#define HEX(n)             \
  ('0' + ((n)>>28 & 0xF)), \
  ('0' + ((n)>>24 & 0xF)), \
  ('0' + ((n)>>20 & 0xF)), \
  ('0' + ((n)>>16 & 0xF)), \
  ('0' + ((n)>>12 & 0xF)), \
  ('0' + ((n)>>8  & 0xF)), \
  ('0' + ((n)>>4  & 0xF)), \
  ('0' + ((n)     & 0xF))

/* Construct a string literal encoding the version number. */
#ifdef COMPILER_VERSION
char const* info_version = "INFO" ":" "compiler_version[" COMPILER_VERSION "]";

/* Construct a string literal encoding the version number components. */
#elif defined(COMPILER_VERSION_MAJOR)
char const info_version[] = {
  'I', 'N', 'F', 'O', ':',
  'c','o','m','p','i','l','e','r','_','v','e','r','s','i','o','n','[',
  COMPILER_VERSION_MAJOR,
# ifdef COMPILER_VERSION_MINOR
  '.', COMPILER_VERSION_MINOR,
#  ifdef COMPILER_VERSION_PATCH
   '.', COMPILER_VERSION_PATCH,
#   ifdef COMPILER_VERSION_TWEAK
    '.', COMPILER_VERSION_TWEAK,
#   endif
#  endif
# endif
  ']','\0'};
#endif

/* Construct a string literal encoding the internal version number. */
#ifdef COMPILER_VERSION_INTERNAL
char const info_version_internal[] = {
  'I', 'N', 'F', 'O', ':',
  'c','o','m','p','i','l','e','r','_','v','e','r','s','i','o','n','_',
  'i','n','t','e','r','n','a','l','[',
  COMPILER_VERSION_INTERNAL,']','\0'};
#elif defined(COMPILER_VERSION_INTERNAL_STR)
char const* info_version_internal = "INFO" ":" "compiler_version_internal[" COMPILER_VERSION_INTERNAL_STR "]";
#endif

/* Construct a string literal encoding the version number components. */
#ifdef SIMULATE_VERSION_MAJOR
char const info_simulate_version[] = {
  'I', 'N', 'F', 'O', ':',
  's','i','m','u','l','a','t','e','_','v','e','r','s','i','o','n','[',
  SIMULATE_VERSION_MAJOR,
# ifdef SIMULATE_VERSION_MINOR
  '.', SIMULATE_VERSION_MINOR,
#  ifdef SIMULATE_VERSION_PATCH
   '.', SIMULATE_VERSION_PATCH,
#   ifdef SIMULATE_VERSION_TWEAK
    '.', SIMULATE_VERSION_TWEAK,
#   endif
#  endif
# endif
  ']','\0'};
#endif

/* Construct the string literal in pieces to prevent the source from
   getting matched.  Store it in a pointer rather than an array
   because some compilers will just produce instructions to fill the
   array rather than assigning a pointer to a static array.  */
char const* info_platform = "INFO" ":" "platform[" PLATFORM_ID "]";
char const* info_arch = "INFO" ":" "arch[" ARCHITECTURE_ID "]";



#define CXX_STD_98 199711L
#define CXX_STD_11 201103L
#define CXX_STD_14 201402L
#define CXX_STD_17 201703L
#define CXX_STD_20 202002L
#define CXX_STD_23 202302L

#if defined(__INTEL_COMPILER) && defined(_MSVC_LANG)
#  if _MSVC_LANG > CXX_STD_17
#    define CXX_STD _MSVC_LANG
#  elif _MSVC_LANG == CXX_STD_17 && defined(__cpp_aggregate_paren_init)
#    define CXX_STD CXX_STD_20
#  elif _MSVC_LANG > CXX_STD_14 && __cplusplus > CXX_STD_17
#    define CXX_STD CXX_STD_20
#  elif _MSVC_LANG > CXX_STD_14
#    define CXX_STD CXX_STD_17
#  elif defined(__INTEL_CXX11_MODE__) && defined(__cpp_aggregate_nsdmi)
#    define CXX_STD CXX_STD_14
#  elif defined(__INTEL_CXX11_MODE__)
#    define CXX_STD CXX_STD_11
#  else
#    define CXX_STD CXX_STD_98
#  endif
#elif defined(_MSC_VER) && defined(_MSVC_LANG)
#  if _MSVC_LANG > __cplusplus
#    define CXX_STD _MSVC_LANG
#  else
#    define CXX_STD __cplusplus
#  endif
#elif defined(__NVCOMPILER)
#  if __cplusplus == CXX_STD_17 && defined(__cpp_aggregate_paren_init)
#    define CXX_STD CXX_STD_20
#  else
#    define CXX_STD __cplusplus
#  endif
#elif defined(__INTEL_COMPILER) || defined(__PGI)
#  if __cplusplus == CXX_STD_11 && defined(__cpp_namespace_attributes)
#    define CXX_STD CXX_STD_17
#  elif __cplusplus == CXX_STD_11 && defined(__cpp_aggregate_nsdmi)
#    define CXX_STD CXX_STD_14
#  else
#    define CXX_STD __cplusplus
#  endif
#elif (defined(__IBMCPP__) || defined(__ibmxl__)) && defined(__linux__)
#  if __cplusplus == CXX_STD_11 && defined(__cpp_aggregate_nsdmi)
#    define CXX_STD CXX_STD_14
#  else
#    define CXX_STD __cplusplus
#  endif
#elif __cplusplus == 1 && defined(__GXX_EXPERIMENTAL_CXX0X__)
#  define CXX_STD CXX_STD_11
#else
#  define CXX_STD __cplusplus
#endif

const char* info_language_standard_default = "INFO" ":" "standard_default["
#if CXX_STD > CXX_STD_23
  "26"
#elif CXX_STD > CXX_STD_20
  "23"
#elif CXX_STD > CXX_STD_17
  "20"
#elif CXX_STD > CXX_STD_14
  "17"
#elif CXX_STD > CXX_STD_11
  "14"
#elif CXX_STD >= CXX_STD_11
  "11"
#else
  "98"
#endif
"]";

const char* info_language_extensions_default = "INFO" ":" "extensions_default["
#if (defined(__clang__) || defined(__GNUC__) || defined(__xlC__) ||           \
     defined(__TI_COMPILER_VERSION__) || defined(__RENESAS__)) &&             \
  !defined(__STRICT_ANSI__)
  "ON"
#else
  "OFF"
#endif
"]";

/*--------------------------------------------------------------------------*/

int main(int argc, char* argv[])
{
  int require = 0;
  require += info_compiler[argc];
  require += info_platform[argc];
  require += info_arch[argc];
#ifdef COMPILER_VERSION_MAJOR
  require += info_version[argc];
#endif
#if defined(COMPILER_VERSION_INTERNAL) || defined(COMPILER_VERSION_INTERNAL_STR)
  require += info_version_internal[argc];
#endif
#ifdef SIMULATE_ID
  require += info_simulate[argc];
#endif
#ifdef SIMULATE_VERSION_MAJOR
  require += info_simulate_version[argc];
#endif
#if defined(__CRAYXT_COMPUTE_LINUX_TARGET)
  require += info_cray[argc];
#endif
  require += info_language_standard_default[argc];
  require += info_language_extensions_default[argc];
  (void)argv;
  return require;
}

// === C:\Users\Cutie Magic 500\projects\creative\screen-ocr\tests\test_core.cpp ===
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

