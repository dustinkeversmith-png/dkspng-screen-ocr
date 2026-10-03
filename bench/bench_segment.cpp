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
        cls = std::make_unique<ocr::Classifier>(
            *atlas, ocr::FeatureWeights{}, bench::has_flag(argc, argv, "--sorted") ? ocr::SearchMode::Sorted : ocr::SearchMode::Boxes,
            std::stof(bench::arg_value(argc, argv, "--margin", "15")));
        cls->enable_memo(true);
        ocr::RecognizerParams rp;
        rp.topk = std::stoi(bench::arg_value(argc, argv, "--topk", std::to_string(rp.topk)));
        rp.atom_merge_gap = std::stof(bench::arg_value(argc, argv, "--atom-gap", std::to_string(rp.atom_merge_gap)));
    rp.atom_merge_px = std::stoi(bench::arg_value(argc, argv, "--atom-px", std::to_string(rp.atom_merge_px)));
        reader = std::make_unique<ScreenReader>(*cls, P, rp);
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
