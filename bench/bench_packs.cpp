// Detection-pack benchmark.
//
//   ui_states     checkbox / radio crops, checked vs unchecked (train+validation -> test)
//                 * TemplatePack (k-NN over ShapeDescriptors) and the training-free CheckStatePack
//   rico_widgets  Rico widget / icon crops, type classification with a screen-disjoint split
//                 (even Rico row -> train, odd row -> test)
//
//   bench_packs [--data datasets] [--save data/packs]   saves the trained packs as .dkpk
#include <cstdio>
#include <map>
#include <string>

#include "common.hpp"
#include "dks/detection/detection_pack.hpp"
#include "dks/detection/rule_packs.hpp"

using namespace dks;
using namespace dks::detect;

namespace {

struct Item {
    std::string label, split;
    Gray8 luma;
};

std::vector<Item> load(const std::string& dir, bool rico) {
    std::vector<Item> out;
    for (const auto& s : bench::load_manifest(dir)) {
        const auto gt = bench::load_gt(s.gt);
        if (gt.empty()) continue;
        win32::Frame f = win32::load_image(s.image);
        if (f.empty()) continue;
        Item it;
        it.label = gt[0].label;
        if (rico) {  // .../r01234_05.png -> Rico row 1234
            const size_t p = s.image.find_last_of("/\\");
            const int row = std::atoi(s.image.c_str() + p + 2);
            it.split = row % 2 == 0 ? "train" : "test";
        } else {
            it.split = s.image.find("/test/") != std::string::npos ? "test" : "train";
        }
        it.luma = to_luma(f.view());
        out.push_back(std::move(it));
    }
    return out;
}

struct Score {
    std::map<std::string, std::map<std::string, int>> conf;  // gt -> pred -> n
    int n = 0, ok = 0;
    void add(const std::string& gt, const std::string& pred) { ++conf[gt][pred], ++n, ok += gt == pred; }
    void print(const char* title) const {
        std::printf("  %-34s accuracy %.3f  (%d / %d)\n", title, n ? double(ok) / n : 0.0, ok, n);
        for (const auto& [gt, row] : conf) {
            int tot = 0, hit = 0;
            for (const auto& [p, c] : row) tot += c, hit += p == gt ? c : 0;
            std::string top;
            for (const auto& [p, c] : row)
                if (p != gt && c * 10 >= tot) top += " " + p + ":" + std::to_string(c);
            std::printf("      %-24s %4d  recall %.2f%s%s\n", gt.c_str(), tot, tot ? double(hit) / tot : 0.0,
                        top.empty() ? "" : "   confused with", top.c_str());
        }
    }
};

}  // namespace

int main(int argc, char** argv) {
    const std::string root = bench::arg_value(argc, argv, "--data", "datasets");
    const std::string save = bench::arg_value(argc, argv, "--save", "data/packs");
    CreateDirectoryA(save.c_str(), nullptr);

    // ---------------------------------------------------------------- checkbox / radio state
    {
        auto items = load(root + "/ui_states", false);
        std::printf("\n== ui_states  (%zu crops)\n", items.size());
        auto pack = std::make_shared<TemplatePack>("check_state.knn", "widget.state");
        for (const auto& it : items)
            if (it.split == "train") pack->add_example(it.label.substr(it.label.rfind('.') + 1), it.luma.cview());
        const CheckStatePack rule;
        Score knn, rl;
        bench::Timer t;
        for (const auto& it : items) {
            if (it.split != "test") continue;
            const std::string gt = it.label.substr(it.label.rfind('.') + 1);
            Patch p;
            p.luma = it.luma.cview();
            p.box = Rect{0, 0, it.luma.width(), it.luma.height()};
            const auto m = pack->classify(p);
            knn.add(gt, m.empty() ? "?" : m.front().tag);
            const auto r = rule.classify(p);
            rl.add(gt, r.empty() ? "?" : r.front().tag);
        }
        std::printf("  %zu exemplars, %.2f ms per crop (both packs)\n", pack->size(), t.ms() / double(knn.n ? knn.n : 1));
        knn.print("TemplatePack k-NN (trained)");
        rl.print("CheckStatePack rule (no training)");
        pack->save(save + "/check_state.dkpk");
    }

    // ---------------------------------------------------------------- Rico widget / icon type
    {
        auto items = load(root + "/rico_widgets", true);
        std::printf("\n== rico_widgets  (%zu crops, screen-disjoint split)\n", items.size());
        if (!items.empty()) {
            auto pack = std::make_shared<TemplatePack>("rico_widget_type.knn", "widget", PackScope{}, std::stoi(bench::arg_value(argc, argv, "--k", "3")));
            for (const auto& it : items)
                if (it.split == "train") pack->add_example(it.label, it.luma.cview());
            Score s, coarse;
            auto group = [](const std::string& l) { return l.rfind("icon.", 0) == 0 ? std::string("icon") : l; };
            for (const auto& it : items) {
                if (it.split != "test") continue;
                Patch p;
                p.luma = it.luma.cview();
                p.box = Rect{0, 0, it.luma.width(), it.luma.height()};
                const auto m = pack->classify(p);
                const std::string pred = m.empty() ? "?" : m.front().tag;
                s.add(it.label, pred);
                coarse.add(group(it.label), group(pred));
            }
            std::printf("  %zu exemplars, %zu classes\n", pack->size(), pack->classes().size());
            s.print("TemplatePack k-NN, fine classes");
            coarse.print("same, icons pooled into 'icon'");
            pack->save(save + "/rico_widget_type.dkpk");
        }
    }
    return 0;
}
