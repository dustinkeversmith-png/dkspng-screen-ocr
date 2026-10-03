// Compile a labelled crop dataset into a detection pack (.dkpk).
//
//   train_pack --classes DIR  --name NAME --tag-key KEY --out data/packs/NAME.dkpk [--k 3] [--holdout 5]
//       DIR/<label>/*.png|jpg|bmp|gif|webp     one sub-folder per class, folder name = label
//   train_pack --manifest DIR ...
//       DIR/manifest.tsv + *.gt.tsv             the repo's dataset format; label = column 5 of the first row
//
// --holdout N   evaluate first: every N-th crop of each class (deterministic) is held out, accuracy and a
//               confusion summary are printed, then the pack is trained on *all* crops and saved.
// See docs/PACKS.md.
#include <algorithm>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

#include "common.hpp"
#include "dks/detection/detection_pack.hpp"

using namespace dks;

namespace {

struct Item {
    std::string label, path;
    Gray8 luma;
};

std::vector<std::string> list_dir(const std::string& dir, bool dirs) {
    std::vector<std::string> out;
    WIN32_FIND_DATAA fd;
    HANDLE h = FindFirstFileA((dir + "/*").c_str(), &fd);
    for (BOOL ok = h != INVALID_HANDLE_VALUE; ok; ok = FindNextFileA(h, &fd)) {
        const std::string n = fd.cFileName;
        if (n == "." || n == "..") continue;
        const bool is_dir = (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0;
        if (is_dir == dirs) out.push_back(n);
    }
    if (h != INVALID_HANDLE_VALUE) FindClose(h);
    std::sort(out.begin(), out.end());  // deterministic order
    return out;
}

bool is_image(const std::string& n) {
    std::string e = n.substr(n.find_last_of('.') + 1);
    std::transform(e.begin(), e.end(), e.begin(), [](unsigned char c) { return char(std::tolower(c)); });
    return e == "png" || e == "jpg" || e == "jpeg" || e == "bmp" || e == "gif" || e == "webp" || e == "tif" || e == "tiff";
}

}  // namespace

int main(int argc, char** argv) {
    const std::string classes = bench::arg_value(argc, argv, "--classes", "");
    const std::string manifest = bench::arg_value(argc, argv, "--manifest", "");
    const std::string name = bench::arg_value(argc, argv, "--name", "my_pack");
    const std::string key = bench::arg_value(argc, argv, "--tag-key", "widget");
    const std::string out = bench::arg_value(argc, argv, "--out", "data/packs/" + name + ".dkpk");
    const int k = std::stoi(bench::arg_value(argc, argv, "--k", "3"));
    const int holdout = std::stoi(bench::arg_value(argc, argv, "--holdout", "0"));
    if (classes.empty() == manifest.empty()) {
        std::fprintf(stderr, "usage: train_pack (--classes DIR | --manifest DIR) --name NAME --tag-key KEY [--out FILE] [--k 3] [--holdout 5]\n");
        return 2;
    }

    std::vector<Item> items;
    auto add = [&](const std::string& label, const std::string& path) {
        win32::Frame f = win32::load_image(path);
        if (f.empty()) { std::fprintf(stderr, "  ! cannot read %s\n", path.c_str()); return; }
        items.push_back({label, path, to_luma(f.view())});
    };
    if (!classes.empty()) {
        for (const auto& label : list_dir(classes, true))
            for (const auto& file : list_dir(classes + "/" + label, false))
                if (is_image(file)) add(label, classes + "/" + label + "/" + file);
    } else {
        for (const auto& s : bench::load_manifest(manifest)) {
            const auto gt = bench::load_gt(s.gt);
            if (!gt.empty()) add(gt[0].label, s.image);
        }
    }
    std::map<std::string, int> per_class;
    for (const auto& it : items) ++per_class[it.label];
    std::printf("%zu crops, %zu classes\n", items.size(), per_class.size());
    for (const auto& [l, c] : per_class) std::printf("  %-28s %5d%s\n", l.c_str(), c, c < 5 ? "   (few examples: expect low recall)" : "");
    if (items.empty()) return 1;

    if (holdout > 1) {
        detect::TemplatePack eval(name, key, {}, k);
        std::map<std::string, int> seen;
        std::vector<const Item*> test;
        for (const auto& it : items)
            if (++seen[it.label] % holdout == 0) test.push_back(&it);
            else eval.add_example(it.label, it.luma.cview());
        std::map<std::string, std::map<std::string, int>> conf;
        int ok = 0;
        for (const Item* it : test) {
            const auto m = eval.classify(detect::describe(it->luma.cview()));
            const std::string pred = m.empty() ? "?" : m.front().tag;
            ++conf[it->label][pred];
            ok += pred == it->label;
        }
        std::printf("\nhold-out (every %d-th crop per class): accuracy %.3f (%d / %zu)\n", holdout,
                    test.empty() ? 0.0 : double(ok) / double(test.size()), ok, test.size());
        for (const auto& [gt, row] : conf) {
            int tot = 0, hit = 0;
            std::string worst;
            int worst_n = 0;
            for (const auto& [p, c] : row) {
                tot += c, hit += p == gt ? c : 0;
                if (p != gt && c > worst_n) worst = p, worst_n = c;
            }
            std::printf("  %-28s recall %.2f  (%d/%d)%s%s\n", gt.c_str(), double(hit) / double(tot), hit, tot,
                        worst.empty() ? "" : "   most confused with ", worst.c_str());
        }
    }

    detect::TemplatePack pack(name, key, {}, k);
    for (const auto& it : items) pack.add_example(it.label, it.luma.cview());
    if (!pack.save(out)) { std::fprintf(stderr, "cannot write %s\n", out.c_str()); return 1; }
    std::printf("\nwrote %s  (%zu exemplars, %zu classes, k=%d, tag key '%s')\n", out.c_str(), pack.size(), pack.classes().size(), k,
                key.c_str());
    return 0;
}
