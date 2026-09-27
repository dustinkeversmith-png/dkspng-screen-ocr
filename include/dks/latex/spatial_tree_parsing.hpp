#pragma once
// Spatial Tree Parsing (deterministic operator dominance) — Layer 3 of the LaTeX stage.
//
// Once math symbols and their boxes are extracted (MathReader), the 2D layout is folded into a tree,
// innermost structures first:
//   1. Radicals   bounding-box containment: every symbol right of the radical's vinculum start and
//                 inside its box is the radicand -> \sqrt { ... }
//   1b. Accents   \bar \tilde \hat \vec \dot: a small mark directly over exactly one glyph
//   2. Fractions  horizontal rules (w/h >= 3, fill >= 0.7), narrowest first: symbols whose x-span lies
//                 within the bar's x-span and sit directly above / below form numerator / denominator ->
//                 \frac { num } { den }. Rule with content only above -> \overline; nothing -> minus.
//   3. Limits     big operators (∑ ∏) take symbols stacked above / below within their x-span.
//   4. Scripts    the remaining nodes are read left to right against the main line (median scale S and
//                 baseline B of the full-size symbols). Each symbol's scale and baseline come from its best
//                 template's metrics, so x-height letters, ascenders and descenders are comparable. A
//                 smaller symbol whose baseline is raised is a superscript, lowered a subscript; runs of
//                 script symbols are parsed recursively (nested scripts, fractions inside scripts).
// Output uses im2latex-style spaced tokens: "x _ { 1 } ^ { 2 }", "\frac { a } { b }".
#include <algorithm>
#include <string>
#include <vector>

#include "math_reader.hpp"

namespace dks::latex {

struct ParseParams {
    float script_scale = 0.84f;   // scale / main scale below this = script size
    float sup_raise = 0.22f;      // baseline raised by > this * S  -> superscript
    float sub_drop = 0.12f;       // baseline lowered by > this * S -> subscript
    float stack_reach = 2.2f;     // numerator / denominator / limits within this * S of the bar / operator
};

namespace detail {

struct Node {
    Rect box;
    std::string tex;
    MathSymbol::Kind kind = MathSymbol::Kind::Glyph;
    char32_t ch = 0;
    float scale = 0, baseline = 0;
    int32_t radicand_x = 0;
    bool group = false;  // composite (fraction / radical / limits): always main line
};

inline float main_scale(const std::vector<Node>& v) {
    std::vector<float> s;
    for (const auto& n : v)
        if (n.scale > 0 && n.kind == MathSymbol::Kind::Glyph) s.push_back(n.scale);
    if (s.empty())
        for (const auto& n : v)
            if (n.scale > 0) s.push_back(n.scale);
    if (s.empty()) return 0;
    std::sort(s.begin(), s.end(), std::greater<float>());
    const size_t k = std::max<size_t>(1, (s.size() * 6 + 9) / 10);  // median of the larger 60%
    return s[(k - 1) / 2];
}

inline std::string parse_nodes(std::vector<Node> v, const ParseParams& P, int depth);

inline std::string join(const std::vector<Node>& v, const ParseParams& P, int depth) {
    return parse_nodes(v, P, depth + 1);
}

inline bool centre_within(const Rect& r, const Rect& span, int slack) {
    const int32_t c = r.x + r.w / 2;
    return c >= span.x - slack && c < span.right() + slack;
}

inline std::string parse_nodes(std::vector<Node> v, const ParseParams& P, int depth) {
    if (v.empty() || depth > 12) return "";
    // A sub-expression (script, numerator, denominator, radicand) that is one one-like stem is "1".
    if (depth > 0 && v.size() == 1 && !v[0].group && v[0].kind == MathSymbol::Kind::Glyph &&
        (v[0].ch == U'l' || v[0].ch == U'I' || v[0].ch == U'J' || v[0].ch == 0x131))
        return "1";
    float S = main_scale(v);
    if (S <= 0) S = 10.f;

    // 1. radicals, smallest first
    for (;;) {
        int best = -1;
        for (int i = 0; i < int(v.size()); ++i)
            if (!v[size_t(i)].group && v[size_t(i)].kind == MathSymbol::Kind::Radical &&
                (best < 0 || v[size_t(i)].box.area() < v[size_t(best)].box.area()))
                best = i;
        if (best < 0) break;
        const Node R = v[size_t(best)];
        std::vector<Node> inside, rest;
        for (int i = 0; i < int(v.size()); ++i) {
            if (i == best) continue;
            const Node& n = v[size_t(i)];
            const int32_t cx = n.box.x + n.box.w / 2, cy = n.box.y + n.box.h / 2;
            (cx >= R.radicand_x && R.box.contains_point(cx, cy) ? inside : rest).push_back(n);
        }
        Node g = R;
        g.group = true;
        g.tex = "\\sqrt { " + join(inside, P, depth) + " }";
        for (const auto& n : inside) g.box = g.box.unite(n.box);
        g.scale = S;
        g.baseline = float(R.box.bottom()) - 0.1f * S;
        rest.push_back(g);
        v.swap(rest);
    }

    // 1b. accents: a short rule / tilde / hat / arrow / dot directly above exactly one glyph, with
    //     nothing stacked above it (so a fraction bar over a one-glyph denominator is never taken)
    for (;;) {
        bool changed = false;
        for (size_t a = 0; a < v.size() && !changed; ++a) {
            const Node& acc = v[a];
            if (acc.group) continue;
            const char* cmd = nullptr;
            if (acc.kind == MathSymbol::Kind::Bar) cmd = "\\bar";
            else if (acc.ch == U'~' || acc.ch == 0x223C) cmd = "\\tilde";
            else if (acc.ch == U'^' || acc.ch == 0x02C6 || acc.ch == 0x2227) cmd = "\\hat";
            else if (acc.ch == 0x2192 || acc.ch == 0x21C0) cmd = "\\vec";
            else if (acc.ch == 0x22C5 || acc.ch == 0x00B7 || acc.ch == U'.') cmd = "\\dot";
            if (!cmd || float(acc.box.h) > 0.45f * S) continue;
            int under = -1, count = 0;
            for (size_t b = 0; b < v.size(); ++b) {
                if (b == a) continue;
                const Node& g = v[b];
                if (g.kind != MathSymbol::Kind::Glyph || g.group) continue;
                if (!centre_within(acc.box, g.box, 1) || g.box.y < acc.box.bottom() - 1) continue;
                if (float(g.box.y - acc.box.bottom()) > 0.35f * S) continue;
                if (float(acc.box.w) > 1.6f * float(g.box.w) + 2.f) continue;
                under = int(b), ++count;
            }
            if (count != 1) continue;
            bool above = false;
            for (size_t b = 0; b < v.size() && !above; ++b)
                above = b != a && centre_within(v[b].box, acc.box, 1) && v[b].box.bottom() <= acc.box.y + 1 &&
                        float(acc.box.y - v[b].box.bottom()) <= P.stack_reach * S;
            if (above) continue;
            Node g = v[size_t(under)];
            g.tex = std::string(cmd) + " { " + g.tex + " }";
            g.box = g.box.unite(acc.box);
            std::vector<Node> rest;
            for (size_t b = 0; b < v.size(); ++b)
                if (b != a && b != size_t(under)) rest.push_back(v[b]);
            rest.push_back(g);
            v.swap(rest);
            changed = true;
        }
        if (!changed) break;
    }

    // 2. fraction bars, narrowest first (inner fractions become groups before the outer bar looks)
    for (;;) {
        int best = -1;
        for (int i = 0; i < int(v.size()); ++i) {
            const Node& n = v[size_t(i)];
            if (n.group || n.kind != MathSymbol::Kind::Bar) continue;
            // only bars that actually have something stacked over or under them
            bool stacked = false;
            for (const auto& o : v)
                if (&o != &n && centre_within(o.box, n.box, 1) && o.box.w <= n.box.w + 2 &&
                    (o.box.bottom() <= n.box.y + 1 || o.box.y >= n.box.bottom() - 1) &&
                    std::min(std::abs(o.box.bottom() - n.box.y), std::abs(o.box.y - n.box.bottom())) <= int32_t(P.stack_reach * S))
                    stacked = true;
            if (stacked && (best < 0 || n.box.w < v[size_t(best)].box.w)) best = i;
        }
        if (best < 0) break;
        const Node B = v[size_t(best)];
        std::vector<Node> num, den, rest;
        const int32_t reach = int32_t(P.stack_reach * S * 2.f);
        for (int i = 0; i < int(v.size()); ++i) {
            if (i == best) continue;
            const Node& n = v[size_t(i)];
            const bool in_span = n.box.x >= B.box.x - 2 && n.box.right() <= B.box.right() + 2;
            if (in_span && n.box.bottom() <= B.box.y + 1 && B.box.y - n.box.bottom() <= reach) num.push_back(n);
            else if (in_span && n.box.y >= B.box.bottom() - 1 && n.box.y - B.box.bottom() <= reach) den.push_back(n);
            else rest.push_back(n);
        }
        Node g = B;
        g.group = true;
        if (!num.empty() && !den.empty()) g.tex = "\\frac { " + join(num, P, depth) + " } { " + join(den, P, depth) + " }";
        else if (!num.empty()) g.tex = "\\overline { " + join(num, P, depth) + " }";
        else g.tex = "\\underline { " + join(den, P, depth) + " }";
        for (const auto& n : num) g.box = g.box.unite(n.box);
        for (const auto& n : den) g.box = g.box.unite(n.box);
        g.scale = S;
        g.baseline = float(B.box.y) + 0.5f * float(B.box.h) + 0.37f * S;  // bar sits on the math axis
        rest.push_back(g);
        v.swap(rest);
    }

    // 3. big-operator limits
    for (size_t i = 0; i < v.size(); ++i) {
        Node& op = v[i];
        if (op.group || (op.ch != 0x2211 && op.ch != 0x220F)) continue;
        std::vector<Node> above, below, rest;
        const int32_t reach = int32_t(P.stack_reach * S);
        for (size_t j = 0; j < v.size(); ++j) {
            if (j == i) continue;
            const Node& n = v[j];
            if (centre_within(n.box, op.box, 2) && n.box.bottom() <= op.box.y + 1 && op.box.y - n.box.bottom() <= reach) above.push_back(n);
            else if (centre_within(n.box, op.box, 2) && n.box.y >= op.box.bottom() - 1 && n.box.y - op.box.bottom() <= reach) below.push_back(n);
        }
        if (above.empty() && below.empty()) continue;
        Node g = op;
        g.group = true;
        g.tex = op.tex;
        if (!below.empty()) g.tex += " _ { " + join(below, P, depth) + " }";
        if (!above.empty()) g.tex += " ^ { " + join(above, P, depth) + " }";
        for (const auto& n : above) g.box = g.box.unite(n.box);
        for (const auto& n : below) g.box = g.box.unite(n.box);
        for (size_t j = 0; j < v.size(); ++j) {
            if (j == i) continue;
            bool used = false;
            for (const auto& n : above) used |= n.box == v[j].box;
            for (const auto& n : below) used |= n.box == v[j].box;
            if (!used) rest.push_back(v[j]);
        }
        rest.push_back(g);
        v.swap(rest);
        i = size_t(-1);  // restart: indices changed
    }

    // 4. main line + scripts
    std::sort(v.begin(), v.end(), [](const Node& a, const Node& b) {
        return a.box.x != b.box.x ? a.box.x < b.box.x : a.box.y < b.box.y;
    });
    S = main_scale(v);
    if (S <= 0) S = 10.f;
    std::vector<float> bl;
    for (const auto& n : v)
        if (n.group || n.scale >= P.script_scale * S) bl.push_back(n.baseline);
    std::sort(bl.begin(), bl.end());
    const float B = bl.empty() ? 0.f : bl[bl.size() / 2];

    enum Role { Main, Sup, Sub };
    auto role = [&](const Node& n) {
        if (n.group || n.scale <= 0 || n.scale >= P.script_scale * S) return Main;
        if (n.ch == U'.' || n.ch == U',' || n.ch == U';') return Main;  // punctuation sits on the line
        if (n.baseline < B - P.sup_raise * S) return Sup;
        if (n.baseline > B + P.sub_drop * S) return Sub;
        return Main;
    };
    // "1" context: at 4-5 px a Computer-Modern 1 and a slanted l / I / J are the same stem. A lone
    // one-like glyph that is a whole script group, or that touches a digit on the main line, is "1".
    auto one_like = [](const Node& n) {
        return !n.group && n.kind == MathSymbol::Kind::Glyph && (n.ch == U'l' || n.ch == U'I' || n.ch == U'J' || n.ch == 0x131);
    };
    auto digit = [](const Node& n) { return !n.group && ((n.ch >= U'0' && n.ch <= U'9') || n.tex == "1"); };
    for (size_t i = 0; i < v.size(); ++i) {
        if (!one_like(v[i]) || role(v[i]) != Main) continue;
        for (size_t j : {i - 1, i + 1}) {
            if (j >= v.size() || role(v[j]) != Main || !digit(v[j])) continue;
            const int32_t gap = j < i ? v[i].box.x - v[j].box.right() : v[j].box.x - v[i].box.right();
            if (float(gap) <= 0.35f * S) v[i].tex = "1";
        }
    }
    auto script = [&](const std::vector<Node>& g) {
        if (g.size() == 1 && one_like(g[0])) return std::string("1");
        return join(g, P, depth);
    };

    std::string out;
    auto emit = [&](const std::string& t) {
        if (t.empty()) return;
        if (!out.empty()) out += ' ';
        out += t;
    };
    for (size_t i = 0; i < v.size();) {
        if (role(v[i]) == Main) {
            emit(v[i].tex);
            ++i;
        } else if (out.empty()) {
            emit("{ }");  // pre-script with no base
        }
        std::vector<Node> sup, sub;
        while (i < v.size() && role(v[i]) != Main) (role(v[i]) == Sup ? sup : sub).push_back(v[i]), ++i;
        if (!sub.empty()) emit("_ { " + script(sub) + " }");
        if (!sup.empty()) emit("^ { " + script(sup) + " }");
    }
    return out;
}

}  // namespace detail

inline std::string parse_latex(const std::vector<MathSymbol>& syms, const ParseParams& P = {}) {
    std::vector<detail::Node> v;
    v.reserve(syms.size());
    for (const auto& s : syms) {
        detail::Node n;
        n.box = s.box;
        n.tex = s.tex;
        n.kind = s.kind;
        n.ch = s.ch;
        n.scale = s.scale;
        n.baseline = s.baseline;
        n.radicand_x = s.radicand_x;
        v.push_back(std::move(n));
    }
    return detail::parse_nodes(std::move(v), P, 0);
}

}  // namespace dks::latex
