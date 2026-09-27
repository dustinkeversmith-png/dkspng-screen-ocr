#pragma once
// LaTeX comparison for im2latex-style token strings.
//
// Both sides are canonicalised first, removing notation that does not change the rendered symbols or
// their 2D structure: spacing (\, \; \quad ~ \hspace{..}), sizing (\left \right \big ...), style / font
// switches (\bf \mathrm \cal \displaystyle ...), synonyms (\le -> \leq, \to -> \rightarrow, \lbrack -> [),
// dot spellings (\dots \ldots \cdots ". . .") and braces around a single token ("{ x }" -> "x").
//
//   token edit distance / |gt|, exact match      structure + symbols
//   symbol-bag precision / recall / F1           symbols only (ignores { } ^ _ \frac \sqrt ...)
#include <algorithm>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace dks::eval {

inline std::vector<std::string> latex_canonical(const std::string& s) {
    static const std::set<std::string> drop = {
        "\\,", "\\;", "\\:", "\\!", "\\quad", "\\qquad", "~", "\\", "\\ ", "\\displaystyle", "\\textstyle", "\\scriptstyle",
        "\\scriptscriptstyle", "\\nonumber", "\\big", "\\Big", "\\bigg", "\\Bigg", "\\bigl", "\\bigr", "\\Bigl", "\\Bigr",
        "\\biggl", "\\biggr", "\\Biggl", "\\Biggr", "\\left", "\\right", "\\left.", "\\right.", "\\limits", "\\nolimits",
        "\\bf", "\\rm", "\\it", "\\cal", "\\mathrm", "\\mathcal", "\\mathbf", "\\mathit", "\\mathsf", "\\sf", "\\boldmath",
        "\\operatorname", "\\operatorname*", "\\textrm", "\\mbox", "\\hbox", "\\text", "\\mathtt", "\\tt", "\\emph", "\\mit",
        "\\tiny", "\\small", "\\scriptsize", "\\footnotesize", "\\protect"};
    static const std::map<std::string, std::string> syn = {
        {"\\left(", "("}, {"\\right)", ")"}, {"\\left[", "["}, {"\\right]", "]"}, {"\\left\\{", "\\{"}, {"\\right\\}", "\\}"},
        {"\\left|", "|"}, {"\\right|", "|"}, {"\\left<", "\\langle"}, {"\\right>", "\\rangle"}, {"\\left\\langle", "\\langle"},
        {"\\right\\rangle", "\\rangle"}, {"\\left\\|", "\\Vert"}, {"\\right\\|", "\\Vert"}, {"\\|", "\\Vert"},
        {"\\left\\vert", "|"}, {"\\right\\vert", "|"}, {"\\vert", "|"}, {"\\mid", "|"}, {"\\lbrack", "["}, {"\\rbrack", "]"},
        {"\\lbrace", "\\{"}, {"\\rbrace", "\\}"}, {"\\le", "\\leq"}, {"\\ge", "\\geq"}, {"\\ne", "\\neq"},
        {"\\to", "\\rightarrow"}, {"\\dots", "\\cdots"}, {"\\ldots", "\\cdots"}, {"\\prime", "'"}, {"\\ast", "*"},
        {"\\dag", "\\dagger"}, {"\\sp", "^"}, {"\\sb", "_"}, {"\\over", "\\frac"}, {"\\lgroup", "("}, {"\\rgroup", ")"}};
    std::vector<std::string> t;
    std::istringstream in(s);
    std::string w;
    std::vector<std::string> raw;
    while (in >> w) raw.push_back(w);
    for (size_t i = 0; i < raw.size(); ++i) {
        std::string x = raw[i];
        if (x == "\\hspace" || x == "\\vspace" || x == "\\kern") {  // drop the command and its argument group
            if (i + 1 < raw.size() && raw[i + 1] == "{") {
                int d = 0;
                for (++i; i < raw.size(); ++i) {
                    d += raw[i] == "{" ? 1 : raw[i] == "}" ? -1 : 0;
                    if (d == 0) break;
                }
            }
            continue;
        }
        const auto it = syn.find(x);
        if (it != syn.end()) x = it->second;
        if (drop.count(x)) continue;
        t.push_back(x);
    }
    // ". . ." -> \cdots
    std::vector<std::string> u;
    for (size_t i = 0; i < t.size(); ++i) {
        if (i + 2 < t.size() && t[i] == "." && t[i + 1] == "." && t[i + 2] == ".") {
            u.push_back("\\cdots");
            i += 2;
        } else {
            u.push_back(t[i]);
        }
    }
    // "{ X }" -> "X" and "{ }" -> nothing, until stable
    for (bool changed = true; changed;) {
        changed = false;
        std::vector<std::string> v;
        for (size_t i = 0; i < u.size(); ++i) {
            const bool keep_group = i > 0 && (u[i - 1] == "\\frac" || u[i - 1] == "\\sqrt" || (i > 1 && u[i - 2] == "\\frac"));
            if (u[i] == "{" && i + 1 < u.size() && u[i + 1] == "}" && !keep_group) { ++i; changed = true; continue; }
            if (u[i] == "{" && i + 2 < u.size() && u[i + 2] == "}" && u[i + 1] != "{" && u[i + 1] != "}") {
                v.push_back(u[i + 1]);
                i += 2;
                changed = true;
                continue;
            }
            v.push_back(u[i]);
        }
        u.swap(v);
    }
    return u;
}

inline bool latex_structural(const std::string& t) {
    return t == "{" || t == "}" || t == "^" || t == "_" || t == "\\frac" || t == "\\sqrt" || t == "\\overline" ||
           t == "\\underline" || t == "\\hat" || t == "\\bar" || t == "\\tilde" || t == "\\vec" || t == "\\dot" ||
           t == "\\ddot" || t == "\\widetilde" || t == "\\widehat" || t == "&" || t == "\\\\";
}

struct LatexScore {
    size_t n = 0, exact = 0, edits = 0, tokens = 0;
    size_t sym_tp = 0, sym_pred = 0, sym_gt = 0;

    void add(const std::string& gt, const std::string& pred) {
        const auto g = latex_canonical(gt), p = latex_canonical(pred);
        ++n;
        tokens += g.size();
        // token-level Levenshtein
        std::vector<size_t> prev(p.size() + 1), cur(p.size() + 1);
        for (size_t j = 0; j <= p.size(); ++j) prev[j] = j;
        for (size_t i = 1; i <= g.size(); ++i) {
            cur[0] = i;
            for (size_t j = 1; j <= p.size(); ++j)
                cur[j] = std::min({prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (g[i - 1] == p[j - 1] ? 0 : 1)});
            std::swap(prev, cur);
        }
        edits += prev[p.size()];
        exact += prev[p.size()] == 0;
        std::map<std::string, int> bag;
        for (const auto& x : g)
            if (!latex_structural(x)) ++bag[x], ++sym_gt;
        for (const auto& x : p)
            if (!latex_structural(x)) {
                ++sym_pred;
                if (bag[x] > 0) --bag[x], ++sym_tp;
            }
    }
    double ted() const { return tokens ? double(edits) / double(tokens) : 0; }
    double em() const { return n ? double(exact) / double(n) : 0; }
    double sym_p() const { return sym_pred ? double(sym_tp) / double(sym_pred) : 0; }
    double sym_r() const { return sym_gt ? double(sym_tp) / double(sym_gt) : 0; }
    double sym_f1() const { const double p = sym_p(), r = sym_r(); return p + r > 0 ? 2 * p * r / (p + r) : 0; }
};

}  // namespace dks::eval
