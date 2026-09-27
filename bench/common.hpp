#pragma once
// Shared bench helpers: manifest / GT loading and debug overlays.
#include <chrono>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "dks/core/geometry.hpp"
#include "dks/platform/win32.hpp"

namespace bench {

struct GtBox {
    dks::Rect box;
    std::string label, text;
};

struct Sample {
    std::string image, gt;  // absolute-ish paths
};

inline std::string unescape(const std::string& s) {
    std::string o;
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '\\' && i + 1 < s.size()) {
            const char c = s[++i];
            o.push_back(c == 't' ? '\t' : c == 'n' ? '\n' : c);
        } else o.push_back(s[i]);
    }
    return o;
}

inline std::vector<std::string> split_tabs(const std::string& line) {
    std::vector<std::string> f;
    std::string cur;
    for (char c : line) {
        if (c == '\t') f.push_back(cur), cur.clear();
        else if (c != '\r') cur.push_back(c);
    }
    f.push_back(cur);
    return f;
}

inline std::vector<Sample> load_manifest(const std::string& dir) {
    std::vector<Sample> out;
    std::ifstream in(dir + "/manifest.tsv", std::ios::binary);
    std::string line;
    while (std::getline(in, line)) {
        const auto f = split_tabs(line);
        if (f.size() >= 2) out.push_back({dir + "/" + f[0], dir + "/" + f[1]});
    }
    return out;
}

inline std::vector<GtBox> load_gt(const std::string& path) {
    std::vector<GtBox> out;
    std::ifstream in(path, std::ios::binary);
    std::string line;
    while (std::getline(in, line)) {
        const auto f = split_tabs(line);
        if (f.size() < 5) continue;
        GtBox g;
        g.box = dks::Rect{std::stoi(f[0]), std::stoi(f[1]), std::stoi(f[2]), std::stoi(f[3])};
        g.label = unescape(f[4]);
        g.text = f.size() > 5 ? unescape(f[5]) : "";
        out.push_back(std::move(g));
    }
    return out;
}

struct Timer {
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    double ms() const {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    }
};

inline void draw_rect(dks::win32::Frame& f, const dks::Rect& r, uint8_t R, uint8_t G, uint8_t B, int thick = 1) {
    auto put = [&](int32_t x, int32_t y) {
        if (x < 0 || y < 0 || x >= f.width || y >= f.height) return;
        uint8_t* p = &f.pixels[(size_t(y) * f.width + x) * 4];
        p[0] = B, p[1] = G, p[2] = R, p[3] = 255;
    };
    for (int t = 0; t < thick; ++t) {
        for (int32_t x = r.x - t; x < r.right() + t; ++x) put(x, r.y - t), put(x, r.bottom() - 1 + t);
        for (int32_t y = r.y - t; y < r.bottom() + t; ++y) put(r.x - t, y), put(r.right() - 1 + t, y);
    }
}

inline std::string arg_value(int argc, char** argv, const std::string& key, const std::string& def) {
    for (int i = 1; i + 1 < argc; ++i)
        if (argv[i] == key) return argv[i + 1];
    return def;
}
inline bool has_flag(int argc, char** argv, const std::string& key) {
    for (int i = 1; i < argc; ++i)
        if (argv[i] == key) return true;
    return false;
}

}  // namespace bench
