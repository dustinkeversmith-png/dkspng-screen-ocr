#pragma once
#include <cstdint>
#include <string>
#include <string_view>

namespace dks {

inline std::u32string utf8_decode(std::string_view s) {
    std::u32string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size();) {
        const uint8_t c = uint8_t(s[i]);
        char32_t cp;
        int n;
        if (c < 0x80) cp = c, n = 1;
        else if ((c >> 5) == 0x6) cp = c & 0x1F, n = 2;
        else if ((c >> 4) == 0xE) cp = c & 0x0F, n = 3;
        else if ((c >> 3) == 0x1E) cp = c & 0x07, n = 4;
        else { out.push_back(0xFFFD); ++i; continue; }
        if (i + size_t(n) > s.size()) { out.push_back(0xFFFD); break; }
        for (int k = 1; k < n; ++k) cp = (cp << 6) | (uint8_t(s[i + size_t(k)]) & 0x3F);
        out.push_back(cp);
        i += size_t(n);
    }
    return out;
}

inline void utf8_append(std::string& out, char32_t cp) {
    if (cp < 0x80) out.push_back(char(cp));
    else if (cp < 0x800) out.push_back(char(0xC0 | (cp >> 6))), out.push_back(char(0x80 | (cp & 0x3F)));
    else if (cp < 0x10000)
        out.push_back(char(0xE0 | (cp >> 12))), out.push_back(char(0x80 | ((cp >> 6) & 0x3F))),
            out.push_back(char(0x80 | (cp & 0x3F)));
    else
        out.push_back(char(0xF0 | (cp >> 18))), out.push_back(char(0x80 | ((cp >> 12) & 0x3F))),
            out.push_back(char(0x80 | ((cp >> 6) & 0x3F))), out.push_back(char(0x80 | (cp & 0x3F)));
}

inline std::string utf8_encode(std::u32string_view s) {
    std::string out;
    for (char32_t c : s) utf8_append(out, c);
    return out;
}

}  // namespace dks
