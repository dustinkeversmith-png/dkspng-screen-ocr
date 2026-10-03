#!/usr/bin/env python3
"""
LaTeX stage inspection report.

    build/bench_latex --dump results/latex_test_dump.tsv
    python tools/generate_latex_html_report.py results/latex_test_dump.tsv --out report/latex_report.html [--limit 400]

Input: the TSV written by `bench_latex --dump` (image_path <TAB> ground_truth <TAB> prediction).
Output: one dark-themed HTML file with MathJax 3 (loaded from the jsDelivr CDN): formula thumbnails embedded as
data URIs, ground truth and prediction rendered and raw, token edit distance and status per row, a summary
header, filters, search, and a diagram of the stage architecture.

Scores use the same canonicalisation as include/dks/eval/latex_metrics.hpp (spacing, sizing and font commands,
synonyms and single-token braces removed from both sides), so the header matches bench_latex.
"""
from __future__ import annotations

import argparse
import base64
import html
import json
import os
from pathlib import Path

DROP = set(r"""\, \; \: \! \quad \qquad ~ \ \displaystyle \textstyle \scriptstyle \scriptscriptstyle \nonumber \big \Big
\bigg \Bigg \bigl \bigr \Bigl \Bigr \biggl \biggr \Biggl \Biggr \left \right \left. \right. \limits \nolimits \bf \rm \it
\cal \mathrm \mathcal \mathbf \mathit \mathsf \sf \boldmath \operatorname \operatorname* \textrm \mbox \hbox \text
\mathtt \tt \emph \mit \tiny \small \scriptsize \footnotesize \protect""".split()) | {"\\ "}
SYN = {r"\left(": "(", r"\right)": ")", r"\left[": "[", r"\right]": "]", r"\left\{": r"\{", r"\right\}": r"\}",
       r"\left|": "|", r"\right|": "|", r"\left<": r"\langle", r"\right>": r"\rangle", r"\left\langle": r"\langle",
       r"\right\rangle": r"\rangle", r"\left\|": r"\Vert", r"\right\|": r"\Vert", r"\|": r"\Vert", r"\left\vert": "|",
       r"\right\vert": "|", r"\vert": "|", r"\mid": "|", r"\lbrack": "[", r"\rbrack": "]", r"\lbrace": r"\{",
       r"\rbrace": r"\}", r"\le": r"\leq", r"\ge": r"\geq", r"\ne": r"\neq", r"\to": r"\rightarrow", r"\dots": r"\cdots",
       r"\ldots": r"\cdots", r"\prime": "'", r"\ast": "*", r"\dag": r"\dagger", r"\sp": "^", r"\sb": "_",
       r"\over": r"\frac", r"\lgroup": "(", r"\rgroup": ")"}
STRUCT = {"{", "}", "^", "_", r"\frac", r"\sqrt", r"\overline", r"\underline", r"\hat", r"\bar", r"\tilde", r"\vec",
          r"\dot", r"\ddot", r"\widetilde", r"\widehat", "&", r"\\"}


def canonical(s: str) -> list[str]:
    raw = s.split()
    t, i = [], 0
    while i < len(raw):
        x = raw[i]
        if x in (r"\hspace", r"\vspace", r"\kern"):
            if i + 1 < len(raw) and raw[i + 1] == "{":
                d, i = 0, i + 1
                while i < len(raw):
                    d += 1 if raw[i] == "{" else -1 if raw[i] == "}" else 0
                    if d == 0:
                        break
                    i += 1
            i += 1
            continue
        x = SYN.get(x, x)
        if x not in DROP:
            t.append(x)
        i += 1
    u, i = [], 0
    while i < len(t):
        if t[i:i + 3] == [".", ".", "."]:
            u.append(r"\cdots")
            i += 3
        else:
            u.append(t[i])
            i += 1
    changed = True
    while changed:
        changed, v, i = False, [], 0
        while i < len(u):
            keep_group = i > 0 and (u[i - 1] in (r"\frac", r"\sqrt") or (i > 1 and u[i - 2] == r"\frac"))
            if u[i] == "{" and i + 1 < len(u) and u[i + 1] == "}" and not keep_group:
                i += 2
                changed = True
                continue
            if u[i] == "{" and i + 2 < len(u) and u[i + 2] == "}" and u[i + 1] not in ("{", "}"):
                v.append(u[i + 1])
                i += 3
                changed = True
                continue
            v.append(u[i])
            i += 1
        u = v
    return u


def edit_distance(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def data_uri(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif"}.get(p.suffix.lower(), "image/png")
    return f"data:{mime};base64," + base64.b64encode(p.read_bytes()).decode()


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>dks LaTeX Inspection</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<script>
window.MathJax = {
  loader: { load: ['[tex]/noerrors', '[tex]/noundefined'] },
  tex: { packages: { '[+]': ['noerrors', 'noundefined'] },
         macros: { cal: ['\\mathcal{#1}', 1], bf: ['\\mathbf{#1}', 1], rm: ['\\mathrm{#1}', 1], boldmath: '' } },
  options: { processHtmlClass: 'tex', ignoreHtmlClass: 'page' },
  chtml: { scale: 0.95 }
};
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-mml-chtml.js" async></script>
<style>
:root{color-scheme:dark;--bg:#0f1316;--panel:#161c20;--soft:#1e262b;--rule:#2a343a;--ink:#e3e9ec;--muted:#93a1a8;
--accent:#6cc4ff;--good:#5fcf8e;--bad:#ff7a6b;--warn:#e7b555;--mono:"IBM Plex Mono",Consolas,monospace;
--sans:"IBM Plex Sans",system-ui,"Segoe UI",sans-serif}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.55 var(--sans)}
.page{max-width:1280px;margin:0 auto;padding-inline:18px;padding-block:24px 60px;display:grid;gap:26px}
h1{font-size:28px;margin:0;font-weight:600;letter-spacing:.01em}
h2{font-size:15px;margin:0;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;font-weight:600}
p{margin:0;max-width:75ch;color:var(--muted)}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}
.stat{background:var(--panel);border:1px solid var(--rule);border-radius:6px;padding:12px 14px}
.stat b{display:block;font:500 24px var(--mono);font-variant-numeric:tabular-nums;color:var(--ink)}
.stat span{color:var(--muted);font-size:12.5px}
.arch{background:var(--panel);border:1px solid var(--rule);border-radius:6px;padding:16px;display:grid;gap:12px}
.arch svg{width:100%;height:auto;display:block}
.arch svg text{fill:var(--ink);font:12px var(--sans)}
.arch svg .sub{fill:var(--muted);font-size:11px}
.arch svg rect{fill:var(--soft);stroke:var(--rule)}
.arch svg rect.hl{stroke:var(--accent)}
.arch svg path{stroke:var(--muted);fill:none;stroke-width:1.4}
.tools{display:flex;flex-wrap:wrap;gap:8px;align-items:center}
.tools button{background:var(--panel);color:var(--ink);border:1px solid var(--rule);border-radius:99px;padding:5px 12px;font:500 13px var(--sans);cursor:pointer}
.tools button[aria-pressed=true]{background:var(--ink);color:var(--bg)}
.tools input{background:var(--panel);color:var(--ink);border:1px solid var(--rule);border-radius:6px;padding:6px 10px;font:13px var(--mono);min-width:0;flex:1 1 220px}
.tools select{background:var(--panel);color:var(--ink);border:1px solid var(--rule);border-radius:6px;padding:5px 8px}
.wrap{overflow-x:auto;border:1px solid var(--rule);border-radius:6px}
table{border-collapse:collapse;width:100%;min-width:980px}
th,td{padding:10px;border-bottom:1px solid var(--rule);vertical-align:top;text-align:left}
th{position:sticky;top:0;background:var(--panel);color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.06em;z-index:1}
td.img img{background:#fff;border-radius:3px;max-width:260px;height:auto;display:block}
td.img small{color:var(--muted);font:11px var(--mono);word-break:break-all}
.tex{overflow-x:auto;max-width:360px;padding-bottom:4px}
.raw{font:12px/1.45 var(--mono);color:var(--muted);word-break:break-word;margin-top:6px}
.status{font:600 12px var(--sans);padding:3px 8px;border-radius:99px;white-space:nowrap}
.ok{background:rgba(95,207,142,.15);color:var(--good)}
.mid{background:rgba(231,181,85,.14);color:var(--warn)}
.no{background:rgba(255,122,107,.14);color:var(--bad)}
td.num{font:500 13px var(--mono);font-variant-numeric:tabular-nums;white-space:nowrap}
.note{color:var(--muted);font-size:12.5px}
</style></head>
<body><div class="page">
<header style="display:grid;gap:10px">
<h1>dks LaTeX inspection</h1>
<p>Formula images read by the deterministic LaTeX stage (math atlas + spatial tree parser), compared with the im2latex
ground truth after canonicalisation. Rendered with MathJax; the raw token strings sit underneath each rendering.</p>
</header>
<section class="stats" id="stats"></section>
<section class="arch">
<h2>How a formula is read</h2>
<svg viewBox="0 0 1200 210" role="img" aria-label="Pipeline diagram">
  <rect x="8" y="20" width="160" height="66" rx="6"/><text x="22" y="46">Frame / crop</text><text class="sub" x="22" y="66">BGRA screen or image</text>
  <rect x="200" y="20" width="200" height="66" rx="6"/><text x="214" y="46">Layer 0  LayoutStage</text><text class="sub" x="214" y="66">edge map, run CCL, text boxes</text>
  <rect x="432" y="20" width="200" height="66" rx="6"/><text x="446" y="46">Layer 2  TextReadStage</text><text class="sub" x="446" y="66">ui_fonts.dksa, one baseline</text>
  <rect class="hl" x="664" y="20" width="250" height="66" rx="6"/><text x="678" y="46">LatexStage regions + gate</text><text class="sub" x="678" y="66">boxes not read as linear text</text>
  <rect class="hl" x="664" y="120" width="250" height="78" rx="6"/><text x="678" y="144">MathReader</text><text class="sub" x="678" y="163">math_fonts.dksa, symbol per component,</text><text class="sub" x="678" y="180">scale + baseline per symbol, prior</text>
  <rect class="hl" x="946" y="120" width="246" height="78" rx="6"/><text x="960" y="144">Spatial tree parser</text><text class="sub" x="960" y="163">radicals, accents, fractions,</text><text class="sub" x="960" y="180">limits, scripts -> LaTeX tokens</text>
  <rect x="946" y="20" width="246" height="66" rx="6"/><text x="960" y="46">Region {"latex", tokens}</text><text class="sub" x="960" y="66">in AnalysisContext::regions</text>
  <path d="M168 53 H200 M400 53 H432 M632 53 H664 M789 86 V120 M914 159 H946 M1069 120 V86"/>
</svg>
<p class="note">This benchmark feeds each formula image straight to the MathReader and parser (the blue boxes). On a live
screen the same reader runs inside LatexStage, which first groups text boxes into formula regions and only claims
regions with enough math evidence.</p>
</section>
<section style="display:grid;gap:10px">
<div class="tools">
  <button type="button" id="f-all" data-f="all" aria-pressed="true">All</button>
  <button type="button" id="f-ok" data-f="ok" aria-pressed="false">Exact match</button>
  <button type="button" id="f-no" data-f="no" aria-pressed="false">Mismatch</button>
  <label for="sort" class="note">Sort</label>
  <select id="sort"><option value="file">file order</option><option value="ted">worst first</option><option value="ted-asc">best first</option></select>
  <input id="q" type="search" placeholder="Search LaTeX, e.g. \frac or \alpha">
</div>
<div class="wrap"><table><thead><tr><th>Image</th><th>Ground truth</th><th>Prediction</th><th>Token edits</th></tr></thead><tbody id="rows"></tbody></table></div>
<p class="note" id="shown"></p>
</section>
</div>
<script>
const DATA = __DATA__;
const esc = s => String(s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const S = DATA.summary;
document.getElementById('stats').innerHTML = [
  [S.n.toLocaleString(), 'formulas in the dump'],
  [(100*S.em).toFixed(1)+'%', 'exact match (canonical tokens)'],
  [S.ted.toFixed(3), 'token edit distance / |gt|'],
  [S.sym_f1.toFixed(3), 'symbol F1 (P '+S.sym_p.toFixed(3)+', R '+S.sym_r.toFixed(3)+')'],
  [S.shown.toLocaleString(), 'rows embedded in this page']
].map(([b,s]) => `<div class="stat"><b>${b}</b><span>${s}</span></div>`).join('');
let filter = 'all';
function render() {
  const q = document.getElementById('q').value.trim();
  const sort = document.getElementById('sort').value;
  let rows = DATA.rows.filter(r => filter === 'all' || (filter === 'ok' ? r.exact : !r.exact));
  if (q) rows = rows.filter(r => r.gt.includes(q) || r.pred.includes(q));
  if (sort === 'ted') rows = rows.slice().sort((a,b) => b.ted - a.ted);
  if (sort === 'ted-asc') rows = rows.slice().sort((a,b) => a.ted - b.ted);
  document.getElementById('rows').innerHTML = rows.map(r => {
    const cls = r.exact ? 'ok' : (r.ted <= 0.2 ? 'mid' : 'no');
    const label = r.exact ? 'exact' : (r.ted <= 0.2 ? 'close' : 'mismatch');
    return `<tr><td class="img">${r.img ? `<img src="${r.img}" alt="formula image" loading="lazy">` : ''}<small>${esc(r.name)}</small></td>
      <td><div class="tex">\\[${esc(r.gt)}\\]</div><div class="raw">${esc(r.gt)}</div></td>
      <td><div class="tex">\\[${esc(r.pred)}\\]</div><div class="raw">${esc(r.pred)}</div></td>
      <td class="num"><span class="status ${cls}">${label}</span><br>${r.edits} / ${r.len}<br>${r.ted.toFixed(2)}</td></tr>`;
  }).join('');
  document.getElementById('shown').textContent = `${rows.length} of ${DATA.rows.length} embedded rows shown.`;
  if (window.MathJax && MathJax.typesetPromise) MathJax.typesetPromise([document.getElementById('rows')]).catch(() => {});
}
document.querySelectorAll('.tools button').forEach(b => b.addEventListener('click', () => {
  filter = b.dataset.f;
  document.querySelectorAll('.tools button').forEach(x => x.setAttribute('aria-pressed', String(x === b)));
  render();
}));
document.getElementById('q').addEventListener('input', render);
document.getElementById('sort').addEventListener('change', render);
render();
</script>
</body></html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump", type=Path, help="TSV from `bench_latex --dump`")
    ap.add_argument("--out", type=Path, default=Path("report/latex_report.html"))
    ap.add_argument("--limit", type=int, default=400, help="rows embedded in the page (all rows are scored)")
    args = ap.parse_args()

    n = exact = edits = tokens = sym_tp = sym_pred = sym_gt = 0
    rows = []
    for line in args.dump.read_text(encoding="utf-8").splitlines():
        f = line.split("\t")
        if len(f) < 3:
            continue
        img, gt, pred = f[0], f[1], f[2]
        g, p = canonical(gt), canonical(pred)
        e = edit_distance(g, p)
        n += 1
        exact += e == 0
        edits += e
        tokens += len(g)
        bag = {}
        for x in g:
            if x not in STRUCT:
                bag[x] = bag.get(x, 0) + 1
                sym_gt += 1
        for x in p:
            if x not in STRUCT:
                sym_pred += 1
                if bag.get(x, 0) > 0:
                    bag[x] -= 1
                    sym_tp += 1
        if len(rows) < args.limit:
            rows.append(dict(name=os.path.basename(img), img=data_uri(img), gt=gt, pred=pred, edits=e, len=len(g),
                             ted=e / max(1, len(g)), exact=e == 0))
    sp = sym_tp / max(1, sym_pred)
    sr = sym_tp / max(1, sym_gt)
    summary = dict(n=n, em=exact / max(1, n), ted=edits / max(1, tokens), sym_p=sp, sym_r=sr,
                   sym_f1=2 * sp * sr / max(1e-9, sp + sr), shown=len(rows))
    data = json.dumps(dict(summary=summary, rows=rows), ensure_ascii=False).replace("</", "<\\/")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(PAGE.replace("__DATA__", data), encoding="utf-8")
    print(f"{n} formulas: exact {100 * summary['em']:.1f}%  TED {summary['ted']:.3f}  symbol F1 {summary['sym_f1']:.3f}")
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB, {len(rows)} rows embedded)")


if __name__ == "__main__":
    main()
