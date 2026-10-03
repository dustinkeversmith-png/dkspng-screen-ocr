// Performance benchmark for the screen reading path (layout + OCR of every box), with an output
// checksum so exact optimisations can be verified to change nothing.
//
//   bench_perf [--data datasets] [--threads 1] [--prune 3.0] [--no-memo] [--limit N] [--by-kind]
//
// Each screenshot is read twice in a row: "cold" is the first read, "warm" the same frame again (a
// live screen between changes). The memo cache persists across frames, as it would live.
#include <cstdio>
#include <map>

#include "common.hpp"
#include "dks/pipeline.hpp"

using namespace dks;

int main(int argc, char** argv) {
    const std::string root = bench::arg_value(argc, argv, "--data", "datasets");
    const unsigned threads = unsigned(std::stoul(bench::arg_value(argc, argv, "--threads", "1")));
    const float prune = std::stof(bench::arg_value(argc, argv, "--prune", "3.0"));
    const size_t limit = size_t(std::stoul(bench::arg_value(argc, argv, "--limit", "1000")));
    const bool memo = !bench::has_flag(argc, argv, "--no-memo");

    bench::Timer tl;
    ocr::Atlas atlas;
    if (!atlas.load(bench::arg_value(argc, argv, "--atlas", "data/ui_fonts.dksa"))) return 1;
    atlas.prune(prune);
    const double load_ms = tl.ms();
    bench::Timer tb;
    ocr::Classifier cls(atlas, {}, ocr::SearchMode::Sorted, std::stof(bench::arg_value(argc, argv, "--margin", "15")));
    const double build_ms = tb.ms();
    TextVerifyParams vp;
    vp.read_containers = vp.read_images = bench::has_flag(argc, argv, "--read-all");
    ocr::RecognizerParams rp;
    rp.topk = std::stoi(bench::arg_value(argc, argv, "--topk", std::to_string(rp.topk)));
    rp.atom_merge_gap = std::stof(bench::arg_value(argc, argv, "--atom-gap", std::to_string(rp.atom_merge_gap)));
    rp.atom_merge_px = std::stoi(bench::arg_value(argc, argv, "--atom-px", std::to_string(rp.atom_merge_px)));
    ScreenReader reader(cls, {}, rp, vp, threads);
    reader.enable_memo(memo);
    std::printf("atlas %zu templates (load+prune %.0f ms, index %.0f ms), %u thread(s), memo %s\n", atlas.templates.size(), load_ms,
                build_ms, reader.threads(), memo ? "on" : "off");

    std::vector<win32::Frame> frames;
    for (const auto& s : bench::load_manifest(root + "/zenodo")) {
        if (frames.size() >= limit) break;
        win32::Frame f = win32::load_image(s.image);
        if (!f.empty()) frames.push_back(std::move(f));
    }

    uint64_t checksum = 1469598103934665603ull;
    auto mix = [&](const std::string& s) {
        for (unsigned char c : s) checksum = (checksum ^ c) * 1099511628211ull;
        checksum = (checksum ^ 0xff) * 1099511628211ull;
    };
    struct Acc {
        double layout = 0, ocr = 0, search = 0;
        uint64_t calls = 0, searches = 0, sads = 0, hits = 0;
    } cold, warm;
    std::map<std::string, std::pair<double, uint64_t>> by_kind;  // kind -> (ms, classify calls), cold only
    size_t boxes = 0, verified = 0;
    std::map<std::string, double> phases;
    LayoutParams lp;
    lp.profile = true;
    for (const auto& f : frames) {
        bench::Timer t;
        const Layout L = analyze_layout(f.view(), lp);
        for (const auto& [k, v] : L.timings) phases[k] += v;
        const double lay = t.ms();
        const Gray8 gray = to_luma(f.view());
        for (int pass = 0; pass < 2; ++pass) {
            Acc& a = pass ? warm : cold;
            const auto c0 = reader.counters();
            bench::Timer t2;
            const auto reads = reader.read_boxes(L, gray);
            a.ocr += t2.ms();
            a.layout += lay;
            const auto c1 = reader.counters();
            a.calls += c1.classify_calls - c0.classify_calls, a.searches += c1.search_calls - c0.search_calls;
            a.sads += c1.distance_evals - c0.distance_evals, a.hits += c1.memo_hits - c0.memo_hits;
            a.search += c1.search_ms - c0.search_ms;
            if (pass == 0) {
                boxes += reads.size();
                for (const auto& r : reads) verified += r.is_text, mix(r.text), mix(r.is_text ? "1" : "0");
            }
        }
        if (bench::has_flag(argc, argv, "--by-kind")) {  // separate cold run per kind, memo off
            ScreenReader solo(cls, {}, rp, vp, threads);
            solo.enable_memo(false);
            for (ElementKind k : {ElementKind::Text, ElementKind::Icon, ElementKind::Container, ElementKind::Image}) {
                Layout sub;
                for (const auto& e : L.elements)
                    if (e.kind == k) sub.elements.push_back(e);
                const auto c0 = solo.counters();
                bench::Timer t3;
                solo.read_boxes(sub, gray);
                by_kind[kind_name(k)].first += t3.ms();
                by_kind[kind_name(k)].second += solo.counters().classify_calls - c0.classify_calls;
            }
        }
    }
    const double n = double(frames.size());
    for (int pass = 0; pass < 2; ++pass) {
        const Acc& a = pass ? warm : cold;
        std::printf("%s: layout %5.1f ms | OCR %6.1f ms/frame | classify %6.0f calls, %6.0f searched (memo %4.1f%%), "
                    "search %6.1f ms, %5.1f us/search, %5.0f SAD/search\n",
                    pass ? "warm" : "cold", a.layout / n, a.ocr / n, double(a.calls) / n, double(a.searches) / n,
                    100.0 * double(a.hits) / double(std::max<uint64_t>(1, a.calls)), a.search / n,
                    1000.0 * a.search / double(std::max<uint64_t>(1, a.searches)), double(a.sads) / double(std::max<uint64_t>(1, a.searches)));
    }
    std::printf("%.0f boxes, %.0f verified text per frame\n", double(boxes) / n, double(verified) / n);
    std::printf("layout phases (ms/frame):");
    for (const auto& [k, v] : phases) std::printf(" %s %.1f |", k.c_str(), v / n);
    std::printf("\n");
    for (const auto& [k, v] : by_kind)
        std::printf("  by kind (cold, memo off) %-10s %7.1f ms/frame %7.0f classify calls/frame\n", k.c_str(), v.first / n, double(v.second) / n);
    std::printf("box cache hits: %llu\n", (unsigned long long)reader.box_hits());
    std::printf("reading checksum %016llx\n", (unsigned long long)checksum);
    return 0;
}
