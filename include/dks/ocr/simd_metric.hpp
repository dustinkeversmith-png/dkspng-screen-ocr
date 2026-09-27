#pragma once
// SIMD kernels for the 16x16 glyph metric. Exact integer results, identical on every path
// (AVX2 / SSE2 / scalar), so recognition stays deterministic regardless of the build target.
#include <cstdint>
#include <cstdlib>

#if defined(__AVX2__)
#include <immintrin.h>
#define DKS_SIMD_AVX2 1
#elif defined(__SSE2__) || defined(_M_X64)
#include <emmintrin.h>
#define DKS_SIMD_SSE2 1
#endif

namespace dks::ocr {

// Sum of absolute differences over [begin, end) bytes; begin/end multiples of 32.
inline uint32_t sad_range(const uint8_t* __restrict a, const uint8_t* __restrict b, int begin, int end) noexcept {
#if defined(DKS_SIMD_AVX2)
    __m256i acc = _mm256_setzero_si256();
    for (int i = begin; i < end; i += 32) {
        const __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
        const __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
        acc = _mm256_add_epi64(acc, _mm256_sad_epu8(va, vb));
    }
    const __m128i s = _mm_add_epi64(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
    return uint32_t(_mm_cvtsi128_si32(s)) + uint32_t(_mm_cvtsi128_si32(_mm_srli_si128(s, 8)));
#elif defined(DKS_SIMD_SSE2)
    __m128i acc = _mm_setzero_si128();
    for (int i = begin; i < end; i += 16) {
        const __m128i va = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + i));
        const __m128i vb = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b + i));
        acc = _mm_add_epi64(acc, _mm_sad_epu8(va, vb));
    }
    return uint32_t(_mm_cvtsi128_si32(acc)) + uint32_t(_mm_cvtsi128_si32(_mm_srli_si128(acc, 8)));
#else
    uint32_t s = 0;
    for (int i = begin; i < end; ++i) s += uint32_t(std::abs(int(a[i]) - int(b[i])));
    return s;
#endif
}

inline uint32_t sad256_simd(const uint8_t* __restrict a, const uint8_t* __restrict b) noexcept {
    return sad_range(a, b, 0, 256);
}

// Early-abandon SAD: stops after the first 64-byte quarter whose running sum exceeds `limit`
// (returns that partial sum, which is already > limit, so the caller's rejection is exact).
inline uint32_t sad256_abandon(const uint8_t* __restrict a, const uint8_t* __restrict b, uint32_t limit) noexcept {
    uint32_t s = 0;
    for (int q = 0; q < 256; q += 64) {
        s += sad_range(a, b, q, q + 64);
        if (s > limit) return s;
    }
    return s;
}

}  // namespace dks::ocr
