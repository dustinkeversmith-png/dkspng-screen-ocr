#pragma once
// Minimal non-owning image views + owning planar images, and pixel-format -> luma conversion.
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "geometry.hpp"

namespace dks {

template <class T>
struct ImageView {
    T* data = nullptr;
    int32_t width = 0, height = 0;
    ptrdiff_t stride = 0;  // in elements

    T* row(int32_t y) const noexcept { return data + y * stride; }
    T& at(int32_t x, int32_t y) const noexcept { return data[y * stride + x]; }
    bool empty() const noexcept { return data == nullptr || width <= 0 || height <= 0; }

    ImageView sub(const Rect& r) const noexcept {
        const Rect c = r.clip(width, height);
        return ImageView{data + c.y * stride + c.x, c.w, c.h, stride};
    }
    operator ImageView<const T>() const noexcept { return {data, width, height, stride}; }
};

template <class T>
class Image {
public:
    Image() = default;
    Image(int32_t w, int32_t h, T fill = T{}) : w_(w), h_(h), buf_(size_t(w) * size_t(h), fill) {}

    void reset(int32_t w, int32_t h, T fill = T{}) {
        w_ = w;
        h_ = h;
        buf_.assign(size_t(w) * size_t(h), fill);
    }

    int32_t width() const noexcept { return w_; }
    int32_t height() const noexcept { return h_; }
    T* data() noexcept { return buf_.data(); }
    const T* data() const noexcept { return buf_.data(); }
    T* row(int32_t y) noexcept { return buf_.data() + size_t(y) * w_; }
    const T* row(int32_t y) const noexcept { return buf_.data() + size_t(y) * w_; }
    T& at(int32_t x, int32_t y) noexcept { return buf_[size_t(y) * w_ + x]; }
    const T& at(int32_t x, int32_t y) const noexcept { return buf_[size_t(y) * w_ + x]; }

    ImageView<T> view() noexcept { return {buf_.data(), w_, h_, w_}; }
    ImageView<const T> view() const noexcept { return {buf_.data(), w_, h_, w_}; }
    ImageView<const T> cview() const noexcept { return view(); }

private:
    int32_t w_ = 0, h_ = 0;
    std::vector<T> buf_;
};

using Gray8 = Image<uint8_t>;
using GrayView = ImageView<const uint8_t>;

enum class PixelFormat : uint8_t { Gray8, RGB24, BGR24, RGBA32, BGRA32 };

constexpr int bytes_per_pixel(PixelFormat f) noexcept {
    switch (f) {
        case PixelFormat::Gray8: return 1;
        case PixelFormat::RGB24:
        case PixelFormat::BGR24: return 3;
        default: return 4;
    }
}

// Interleaved colour frame (e.g. a screen capture). Non-owning.
struct ColorView {
    const uint8_t* data = nullptr;
    int32_t width = 0, height = 0;
    ptrdiff_t stride_bytes = 0;
    PixelFormat format = PixelFormat::BGRA32;

    const uint8_t* row(int32_t y) const noexcept { return data + y * stride_bytes; }
    // Channel offsets of R, G, B inside one pixel.
    void rgb_offsets(int& r, int& g, int& b) const noexcept {
        switch (format) {
            case PixelFormat::RGB24:
            case PixelFormat::RGBA32: r = 0, g = 1, b = 2; break;
            case PixelFormat::BGR24:
            case PixelFormat::BGRA32: r = 2, g = 1, b = 0; break;
            default: r = g = b = 0; break;
        }
    }
};

// Rec.601 luma in 8.8 fixed point: Y = (77R + 150G + 29B + 128) >> 8. Loop is branch-free and
// auto-vectorises for the fixed-format inner loops.
inline void to_luma(const ColorView& src, ImageView<uint8_t> dst) noexcept {
    assert(dst.width == src.width && dst.height == src.height);
    const int bpp = bytes_per_pixel(src.format);
    if (src.format == PixelFormat::Gray8) {
        for (int32_t y = 0; y < src.height; ++y) {
            const uint8_t* s = src.row(y);
            uint8_t* d = dst.row(y);
            for (int32_t x = 0; x < src.width; ++x) d[x] = s[x];
        }
        return;
    }
    int ro, go, bo;
    src.rgb_offsets(ro, go, bo);
    for (int32_t y = 0; y < src.height; ++y) {
        const uint8_t* s = src.row(y);
        uint8_t* d = dst.row(y);
        for (int32_t x = 0; x < src.width; ++x, s += bpp)
            d[x] = uint8_t((77u * s[ro] + 150u * s[go] + 29u * s[bo] + 128u) >> 8);
    }
}

inline Gray8 to_luma(const ColorView& src) {
    Gray8 out(src.width, src.height);
    to_luma(src, out.view());
    return out;
}

inline Gray8 crop(GrayView src, const Rect& r) {
    const Rect c = r.clip(src.width, src.height);
    Gray8 out(c.w, c.h);
    for (int32_t y = 0; y < c.h; ++y) {
        const uint8_t* s = src.row(c.y + y) + c.x;
        uint8_t* d = out.row(y);
        for (int32_t x = 0; x < c.w; ++x) d[x] = s[x];
    }
    return out;
}

}  // namespace dks
