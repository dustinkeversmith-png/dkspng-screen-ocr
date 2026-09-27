#pragma once
// dks — deterministic screen segmentation + OCR. Header-only, C++17, no dependencies.
//
//   core/      geometry, image views, UTF-8
//   imgproc/   edge mask, Otsu ink extraction
//   segment/   run-based CCL, containment forest, layout (words / lines / icons / containers)
//   ocr/       glyph features, font atlas, k-NN / VP-tree classifier, word & line recognizer
//   eval/      detection / grouping / hierarchy / CER metrics
//   platform/  (opt-in) Win32 capture + WIC image IO — include explicitly
#include "core/geometry.hpp"
#include "core/image.hpp"
#include "core/utf8.hpp"
#include "eval/metrics.hpp"
#include "imgproc/binarize.hpp"
#include "ocr/ocr.hpp"
#include "segment/ccl.hpp"
#include "segment/hierarchy.hpp"
#include "segment/layout.hpp"
