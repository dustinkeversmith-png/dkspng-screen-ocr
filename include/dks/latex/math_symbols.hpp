#pragma once
// Math atlas labels -> LaTeX. Labels are the codepoints stored in data/math_fonts.dksa
// (tools/build_math_atlas.py): ASCII letters/digits/punctuation, Greek, operators.
#include <string>
#include <unordered_map>

#include "../core/utf8.hpp"

namespace dks::latex {

enum class SymClass : uint8_t { Ordinary, BigOp, Relation, Binary, Open, Close, Punct };

struct SymbolInfo {
    const char* tex;
    SymClass cls;
};

inline const std::unordered_map<char32_t, SymbolInfo>& symbol_table() {
    static const std::unordered_map<char32_t, SymbolInfo> t = [] {
        std::unordered_map<char32_t, SymbolInfo> m;
        const char* greek[] = {"\\alpha", "\\beta", "\\gamma", "\\delta", "\\varepsilon", "\\zeta", "\\eta", "\\theta",
                               "\\iota", "\\kappa", "\\lambda", "\\mu", "\\nu", "\\xi", "o", "\\pi", "\\rho",
                               "\\varsigma", "\\sigma", "\\tau", "\\upsilon", "\\varphi", "\\chi", "\\psi", "\\omega"};
        for (int i = 0; i < 25; ++i) m[char32_t(0x03B1 + i)] = {greek[i], SymClass::Ordinary};
        m[0x03F5] = {"\\epsilon", SymClass::Ordinary};
        m[0x03D1] = {"\\vartheta", SymClass::Ordinary};
        m[0x03D5] = {"\\phi", SymClass::Ordinary};
        m[0x03F1] = {"\\varrho", SymClass::Ordinary};
        m[0x03D6] = {"\\varpi", SymClass::Ordinary};
        const std::pair<char32_t, const char*> upper[] = {{0x0393, "\\Gamma"}, {0x0394, "\\Delta"}, {0x0398, "\\Theta"},
                                                          {0x039B, "\\Lambda"}, {0x039E, "\\Xi"}, {0x03A0, "\\Pi"},
                                                          {0x03A3, "\\Sigma"}, {0x03A5, "\\Upsilon"}, {0x03A6, "\\Phi"},
                                                          {0x03A8, "\\Psi"}, {0x03A9, "\\Omega"}};
        for (auto& [c, s] : upper) m[c] = {s, SymClass::Ordinary};
        const struct { char32_t c; const char* s; SymClass k; } ops[] = {
            {U'+', "+", SymClass::Binary}, {U'-', "-", SymClass::Binary}, {U'=', "=", SymClass::Relation},
            {U'<', "<", SymClass::Relation}, {U'>', ">", SymClass::Relation}, {U'(', "(", SymClass::Open},
            {U')', ")", SymClass::Close}, {U'[', "[", SymClass::Open}, {U']', "]", SymClass::Close},
            {U'{', "\\{", SymClass::Open}, {U'}', "\\}", SymClass::Close}, {U'|', "|", SymClass::Ordinary},
            {U'/', "/", SymClass::Ordinary}, {U',', ",", SymClass::Punct}, {U'.', ".", SymClass::Punct},
            {U';', ";", SymClass::Punct}, {U':', ":", SymClass::Relation}, {U'!', "!", SymClass::Ordinary},
            {U'\'', "'", SymClass::Ordinary}, {U'*', "*", SymClass::Binary}, {U'&', "\\&", SymClass::Ordinary},
            {U'#', "\\#", SymClass::Ordinary}, {U'"', "\"", SymClass::Ordinary}, {U'~', "\\sim", SymClass::Relation},
            {0x00B1, "\\pm", SymClass::Binary}, {0x2213, "\\mp", SymClass::Binary}, {0x00D7, "\\times", SymClass::Binary},
            {0x00F7, "\\div", SymClass::Binary}, {0x00B7, "\\cdot", SymClass::Binary}, {0x22C5, "\\cdot", SymClass::Binary},
            {0x2217, "*", SymClass::Binary}, {0x2218, "\\circ", SymClass::Binary}, {0x2022, "\\bullet", SymClass::Binary},
            {0x22C6, "\\star", SymClass::Binary}, {0x2264, "\\leq", SymClass::Relation}, {0x2265, "\\geq", SymClass::Relation},
            {0x2260, "\\neq", SymClass::Relation}, {0x2248, "\\approx", SymClass::Relation}, {0x2261, "\\equiv", SymClass::Relation},
            {0x223C, "\\sim", SymClass::Relation}, {0x2243, "\\simeq", SymClass::Relation}, {0x2245, "\\cong", SymClass::Relation},
            {0x221D, "\\propto", SymClass::Relation}, {0x226A, "\\ll", SymClass::Relation}, {0x226B, "\\gg", SymClass::Relation},
            {0x221E, "\\infty", SymClass::Ordinary}, {0x2202, "\\partial", SymClass::Ordinary}, {0x2207, "\\nabla", SymClass::Ordinary},
            {0x2211, "\\sum", SymClass::BigOp}, {0x220F, "\\prod", SymClass::BigOp}, {0x222B, "\\int", SymClass::BigOp},
            {0x222E, "\\oint", SymClass::BigOp}, {0x221A, "\\sqrt", SymClass::Ordinary}, {0x2192, "\\rightarrow", SymClass::Relation},
            {0x2190, "\\leftarrow", SymClass::Relation}, {0x2194, "\\leftrightarrow", SymClass::Relation},
            {0x21D2, "\\Rightarrow", SymClass::Relation}, {0x21D4, "\\Leftrightarrow", SymClass::Relation},
            {0x27F6, "\\longrightarrow", SymClass::Relation}, {0x21A6, "\\mapsto", SymClass::Relation},
            {0x2191, "\\uparrow", SymClass::Relation}, {0x2193, "\\downarrow", SymClass::Relation}, {0x2208, "\\in", SymClass::Relation},
            {0x2209, "\\notin", SymClass::Relation}, {0x2282, "\\subset", SymClass::Relation}, {0x2283, "\\supset", SymClass::Relation},
            {0x2286, "\\subseteq", SymClass::Relation}, {0x2287, "\\supseteq", SymClass::Relation}, {0x222A, "\\cup", SymClass::Binary},
            {0x2229, "\\cap", SymClass::Binary}, {0x2200, "\\forall", SymClass::Ordinary}, {0x2203, "\\exists", SymClass::Ordinary},
            {0x00AC, "\\neg", SymClass::Ordinary}, {0x2227, "\\wedge", SymClass::Binary}, {0x2228, "\\vee", SymClass::Binary},
            {0x2297, "\\otimes", SymClass::Binary}, {0x2295, "\\oplus", SymClass::Binary}, {0x2299, "\\odot", SymClass::Binary},
            {0x2020, "\\dagger", SymClass::Ordinary}, {0x2021, "\\ddagger", SymClass::Ordinary}, {0x2016, "\\Vert", SymClass::Ordinary},
            {0x27E8, "\\langle", SymClass::Open}, {0x27E9, "\\rangle", SymClass::Close}, {0x210F, "\\hbar", SymClass::Ordinary},
            {0x2113, "\\ell", SymClass::Ordinary}, {0x2026, "\\ldots", SymClass::Ordinary}, {0x22EF, "\\cdots", SymClass::Ordinary},
            {0x22EE, "\\vdots", SymClass::Ordinary}, {0x2032, "\\prime", SymClass::Ordinary}, {0x22A5, "\\perp", SymClass::Relation},
            {0x2225, "\\parallel", SymClass::Relation}, {0x25B3, "\\triangle", SymClass::Ordinary}, {0x2111, "\\Im", SymClass::Ordinary},
            {0x211C, "\\Re", SymClass::Ordinary}, {0x2118, "\\wp", SymClass::Ordinary}, {0x2135, "\\aleph", SymClass::Ordinary},
            {0x266F, "\\sharp", SymClass::Ordinary}, {0x266D, "\\flat", SymClass::Ordinary}, {0x2294, "\\sqcup", SymClass::Binary},
            {0x2293, "\\sqcap", SymClass::Binary}, {0x22C4, "\\diamond", SymClass::Binary}, {0x21C0, "\\rightharpoonup", SymClass::Relation},
        };
        for (const auto& o : ops) m[o.c] = {o.s, o.k};
        return m;
    }();
    return t;
}

inline std::string tex_of(char32_t c) {
    const auto& t = symbol_table();
    const auto it = t.find(c);
    if (it != t.end()) return it->second.tex;
    std::string s;
    utf8_append(s, c);
    return s;
}

inline SymClass class_of(char32_t c) {
    const auto& t = symbol_table();
    const auto it = t.find(c);
    return it != t.end() ? it->second.cls : SymClass::Ordinary;
}

}  // namespace dks::latex
