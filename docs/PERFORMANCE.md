# Performance study: streaming segmentation and a vector index for the glyph atlas

This note answers three proposals:

1. a single-pass streaming segmentation that accumulates glyph splats during labelling;
2. turning the 16×16 font atlas into a vector search index (projections, quantized or Hamming codes, metric
   trees);
3. a C++17 reference design for both.

Every option was measured, and only one change was kept: an exact **box tree** over the atlas, now
`SearchMode::Boxes` and the default. It makes template search **1.7× faster** with output identical to the old
search, byte for byte.

All numbers come from the 36 Zenodo 1920×1080 screenshots, single thread, using `bench_perf`. The research
queries are 600 to 1,000 real glyph features, recorded from those frames against the 59,582-template UI atlas.

## Where the time goes

Profile of a cold frame before this round (temporary timers, since removed):

| phase | ms/frame | share of OCR |
|---|---:|---:|
| crop ink extraction + per-crop CCL | 1.0 | 0.5 % |
| feature extraction (1,655 features: splat, smoothing, holes, components) | 6.2 | 3.2 % |
| **template search** | **184.6** | **94.7 %** |
| everything else (hypotheses, DP, verification) | 3.0 | 1.5 % |
| *layout (whole screen: edges, CCL, grouping)* | *~15* | *separate* |

A warm frame (the same screen again) costs 0.3 ms of OCR, because the box cache and memo answer it. On a live
desktop, the warm cost is therefore set by layout, and the cold cost by search.

## Task 1: single-pass streaming segmentation and splat accumulation

**Not implemented: the most it can save is about 7 ms of a ~200 ms cold frame (≤ 4 %), and it would cost
accuracy.**

* **Ceiling.** Fusing CCL with ink extraction and splatting can only remove the crop and feature work, about
  7 ms per cold frame. Warm frames never reach that code. Layout CCL is 0.7 ms of the 15 ms layout; the rest is
  edge maps, flat regions and image detection, which a fused pass would not touch.
* **Accuracy cost.** The local streaming threshold would replace per-box ink levels. That ink logic is the most
  accuracy-sensitive part of OCR, and it is decided per box on purpose:
  - Otsu levels set the ink level.
  - A border and run-length vote decides polarity.
  - Shadow mode splits three luminance levels.
  - A 50 % coverage mask cleans the result.

  Local thresholds were measured against it:

  | ink method | ICDAR CER |
  |---|---:|
  | per-box ink (current) | 0.413 |
  | Sauvola | 0.454–0.461 |
  | projection-trimmed Otsu | 0.561 |

  A streaming pass also cannot see the whole box before deciding polarity.
* **What a fused pass would need.** The feature is not a pure function of the run stream. The 16×16 splat is
  normalised by the glyph's final bounding box. Holes and component counts need a second labelling of the
  glyph. Glyph hypotheses span 1–5 cut pieces, and those cuts are decided after labelling. Run sums can be
  streamed. The splat itself cannot be finished until the box is closed, so it would still be a deferred
  pass over the runs.

The sketch in Task 3 shows the run accumulator anyway: it is the right shape if per-frame layout ever becomes
the bottleneck, for example on a 4K screen that changes every frame.

## Task 2: the atlas as a vector index

### The contract to keep

`Classifier::classify` returns the best k = 4 *distinct characters* within best + 15. The metric is:

* L1 over 256 bitmap cells divided by 255,
* plus 14 × |Δ log aspect| + 6 × |Δ holes| + 10 × |Δ components|.

The DP rescoring uses all four candidates, so "same top-1" is not enough. The target is the *identical top-4
list*.

The previous search ("Sorted") computed an exact lower bound for **all** 59.6k templates. The bound uses 4×4
block sums plus the scalar terms, and in a 19-D embedding it is exactly an L1 distance. The search then
refined about 1.3k of them with SAD.

### How small can the candidate set get?

| | mean | median | p90 |
|---|---:|---:|---:|
| templates truly within the final admission radius τ (floor for *any* exact index) | 36 | 18 | 95 |
| templates passing the 4×4 coarse bound | 816 | 401 | — |

### Approximate candidates, then exact re-rank

Each method below picks N candidates with a cheap proxy, then re-ranks them exactly. The figure is the share of
queries whose top-4 list is identical to brute force (1,000 queries):

| method | 64 | 128 | 256 | 512 | 1024 |
|---|---:|---:|---:|---:|---:|
| 4×4 block sums, L1 (the current bound) | 52.9 | 73.8 | 89.3 | 96.2 | 99.1 |
| 8×8 block sums, L1 | **87.4** | **96.5** | **99.4** | 99.7 | 99.7 |
| PCA 32-D, L2 | 40.6 | 56.8 | 74.3 | 89.0 | 95.5 |
| PCA 64-D, L2 | 42.3 | 59.5 | 76.4 | 89.4 | 95.6 |
| 256-bit Hamming code (+ scalar terms) | 62.6 | 79.7 | 90.6 | 96.1 | 98.9 |

* **PCA, DCT and Hadamard:** these are L2-energy projections, and the metric is L1. They are the worst proxies
  here: 64 PCA dimensions rank worse than 16 block sums. Block sums are also *exact* lower bounds (by the
  triangle inequality for L1), and PCA projections are not.
* **Hamming codes:** they discard the grey levels that the smoothed splat relies on.
* **IVF (k-means on 8×8 sums, probing until N candidates):** at k = 2048 and N = 1024 it does 3.4× fewer
  operations, but gets top-1 right on only 99.2 % of queries and the full top-4 on 94.2 %. Every approximate
  index tried changes top-1 on some queries, which fails "maintain accuracy".
* **Look-alike sets (search only characters confusable with the best one):** identical top-4 on only 46–71 %
  of queries.

### Exact metric trees

| index (exact) | evaluated per query | vs flat |
|---|---:|---|
| VP-tree on the full 256-D metric (already in the code) | 15–25k full distances | slower |
| per-(character, font) min/max boxes (an earlier round) | | 818 vs 250 µs: slower |
| k-means in the 19-D bound space, leaf boxes, simulated | 14.6k–39k of 59.6k templates at k = 2048–256 | ≤ 3.6× fewer bounds |

### Kept: the box tree (`SearchMode::Boxes`)

The coarse bound is an L1 distance in 19-D:

* 16 block sums,
* 14 × log aspect,
* 6 × holes,
* 10 × components.

So a min/max box around a group of templates gives an exact lower bound for every member. How it works:

* **Build** (once, ~0.4 s, deterministic). Two-level k-means: about 63 groups, each split into leaves of ~15.
  This gives 4,000 leaves. Templates are stored contiguously, leaf by leaf, in structure-of-arrays form, so a
  leaf's block sums are one 16-lane load per block.
* **Query.**
  1. Compute the box distance of all 4,000 leaves, one dimension at a time (vectorised over leaves).
  2. Sort the leaves within best-box + 12 and visit them nearest first while their box is under τ.
  3. Sweep the remaining leaves once and visit any still under τ.
  4. In a visited leaf, compute member bounds (vectorised across the leaf) and run an early-abandon SAD only
     for members under τ.
* **Exactness.** A leaf is skipped only if its box distance exceeds τ. The box distance never exceeds a
  member's bound, with a 1e-3 slack for float rounding. Ties are broken by template index, so the visit order
  cannot change the result.

| `bench_perf`, cold, 1 thread (3 interleaved runs) | Sorted (before) | **Boxes (now)** |
|---|---:|---:|
| lower bounds per search | 59,582 | **12,561** (4,000 boxes + 8,561 members) |
| full SADs per search | 1,289 | **973** |
| µs per search | 166–168 | **96–97** |
| search, ms/frame | 186–188 | **108** |
| OCR cold, ms/frame | 195–198 | **117–118** |
| reading checksum | `fba7702c4b8e62aa` | `fba7702c4b8e62aa` (identical) |
| index build | 11 ms | 405 ms |
| extra memory | — | 2 × 19 × 4,000 floats ≈ 0.6 MB |

**Cache footprint.** The flat pass streamed about 1.9 MB of block sums plus 0.7 MB of scalars per query; that
is larger than L2 on every desktop CPU, and it ran at memory bandwidth. The box pass reads 0.6 MB of boxes,
then ~8.6k members' 32-byte block sums in ~570 contiguous runs. Bitmaps (256 B each, 15 MB in all) are touched
only for the ~1k SADs, as before.

**Variants tried while tuning:**

| variant | result |
|---|---|
| full sort of every leaf under τ | 898 SADs, but the sort cost ~48 µs: 124 µs/search |
| fully unsorted sweep | 1,426 SADs: 118–124 µs/search |
| near band of 4 / 8 / 12 / 16 / 24 | 12 was best |
| "stop at best + x" (approximate) at x = 0 / 5 | 82 / 100 µs/search before the final ordering, but the readings change |

The approximate stop was dropped: the exact tree reached about the same speed.

## Task 3: C++17 reference design

The index below is what `include/dks/ocr/classifier.hpp` now does (`SearchMode::Boxes`). The streaming
accumulator is a design sketch only (see Task 1); it is not in the library.

### Glyph descriptor and leaf storage (implemented)

```cpp
// 19-D embedding in which the coarse lower bound is plain L1. Block sums stay integers (exact).
struct GlyphVector {
    uint16_t block[16];       // 4x4 block sums of the 16x16 splat (0..4080)
    float aspect, holes, ncomp;  // pre-multiplied by the metric weights (14, 6, 10)
};

// Leaf-major SoA: leaf j owns templates [leaf_start[j], leaf_start[j+1]).
struct BoxIndex {
    std::vector<uint32_t> leaf_start;            // L + 1 entries
    std::vector<float>    box_lo, box_hi;        // [dim][leaf], 19 x L
    std::vector<uint16_t> block_t;               // [block][template]  (leaf-contiguous columns)
    std::vector<float>    aspect, holes, ncomp;  // [template]
    std::vector<uint8_t>  bmp;                   // [template][256]
    std::vector<uint32_t> template_id;           // position -> atlas index (tie-break key)
};
```

### Allocation-free query loop (implemented)

```cpp
// Scratch lives in the per-thread Classifier and is reused across queries, so nothing is allocated.
void search(const GlyphFeature& q, TopChars& top) const {
    const size_t L = leaves();
    float* bd = scratch_bounds(L);
    std::fill_n(bd, L, 0.f);
    for (int e = 0; e < 19; ++e)                     // box distance, vectorised over leaves
        for (size_t j = 0; j < L; ++j)
            bd[j] += scale(e) * std::max({lo(e, j) - q_(e), q_(e) - hi(e, j), 0.f});
    const float band = *std::min_element(bd, bd + L) + 12.f;
    auto& near = scratch_near();                     // capacity retained between queries
    near.clear();
    for (size_t j = 0; j < L; ++j) if (bd[j] <= band) near.push_back(j);
    std::sort(near.begin(), near.end(), by_bound_then_index(bd));
    for (uint32_t j : near) { if (bd[j] > top.tau() + kSlack) break; scan_leaf(q, j, top); }
    for (size_t j = 0; j < L; ++j)                   // any order is exact: tau only shrinks
        if (bd[j] > band && bd[j] <= top.tau() + kSlack) scan_leaf(q, j, top);
}

void scan_leaf(const GlyphFeature& q, size_t j, TopChars& top) const {
    // lb[i] = sum_b |q.block[b] - block_t[b][i]| for every member: 16 vector ops for a leaf of <= 16.
    // Then: bound = lb/255 + scalar terms; if bound <= tau -> early-abandon SAD -> top.offer().
}
```

### Streaming run accumulator (sketch, not implemented)

```cpp
// Fed by the two-row run-length CCL: each horizontal run is merged into its component's accumulator,
// and union-find merges fold one accumulator into the other. Holds only O(1) moments plus a bounded run
// list, because the 16x16 splat needs the final bounding box and so has to be finished in a deferred pass.
struct RunAccumulator {
    int x0 = INT_MAX, y0 = INT_MAX, x1 = -1, y1 = -1;
    uint32_t area = 0;
    uint64_t sum_luma = 0, sum_luma2 = 0;  // for a local ink threshold, if one is ever used
    uint32_t run_begin = 0, run_count = 0; // slice of a frame-wide run arena (no per-component vectors)

    void add_run(int y, int xa, int xb, const uint8_t* row) noexcept;  // xb exclusive
    void merge(const RunAccumulator& o) noexcept;                      // union-find union
};

struct StreamingComponent {
    RunAccumulator acc;
    // Finish: splat the component's runs into 16x16 with the final box, smooth [1 2 1]^2, peak-normalise,
    // compute holes/ncomp from the same runs. Equivalent to sources_feature() on the component's crop,
    // but only for single-component glyphs; multi-piece hypotheses (i, j, %, ':') still need the cut stage.
    GlyphFeature finish(const RunArena& runs) const;
};
```

## Reproducing

```bash
build/bench_perf             # box tree (default)
```

```bash
build/bench_perf --sorted    # the flat scan, for comparison (same checksum)
```

`bench_ocr` and `bench_segment` accept `--sorted` too, and `bench_ocr --vptree` runs the VP-tree.
