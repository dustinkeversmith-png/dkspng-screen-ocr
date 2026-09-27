#pragma once
// Optional Windows platform layer (not included by dks.hpp):
//   capture_screen()  GDI BitBlt of the virtual desktop -> BGRA frame
//   load_image()      WIC decode of PNG/JPEG/GIF/BMP/WebP/TIFF -> BGRA frame
//   save_png()        WIC encode BGRA -> PNG (for debug overlays)
// Link: -lgdi32 -lole32 -lwindowscodecs -luuid
#ifndef _WIN32
#error "dks/platform/win32.hpp is Windows-only"
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <wincodec.h>

#include <cstdint>
#include <string>
#include <vector>

#include "../core/image.hpp"

namespace dks::win32 {

struct Frame {
    std::vector<uint8_t> pixels;  // BGRA, tightly packed
    int32_t width = 0, height = 0;
    ColorView view() const noexcept {
        return ColorView{pixels.data(), width, height, ptrdiff_t(width) * 4, PixelFormat::BGRA32};
    }
    bool empty() const noexcept { return pixels.empty(); }
};

inline std::wstring widen(const std::string& s) {
    if (s.empty()) return {};
    const int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), int(s.size()), nullptr, 0);
    std::wstring w(size_t(n), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.data(), int(s.size()), w.data(), n);
    return w;
}

// Captures the full virtual screen (all monitors). Per-monitor DPI awareness should be enabled by
// the caller (SetProcessDpiAwarenessContext) to get physical pixels.
inline Frame capture_screen() {
    Frame f;
    const int x = GetSystemMetrics(SM_XVIRTUALSCREEN), y = GetSystemMetrics(SM_YVIRTUALSCREEN);
    const int w = GetSystemMetrics(SM_CXVIRTUALSCREEN), h = GetSystemMetrics(SM_CYVIRTUALSCREEN);
    HDC screen = GetDC(nullptr);
    HDC mem = CreateCompatibleDC(screen);
    BITMAPINFO bi{};
    bi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    bi.bmiHeader.biWidth = w;
    bi.bmiHeader.biHeight = -h;  // top-down
    bi.bmiHeader.biPlanes = 1;
    bi.bmiHeader.biBitCount = 32;
    bi.bmiHeader.biCompression = BI_RGB;
    void* bits = nullptr;
    HBITMAP dib = CreateDIBSection(screen, &bi, DIB_RGB_COLORS, &bits, nullptr, 0);
    if (dib && bits) {
        HGDIOBJ old = SelectObject(mem, dib);
        BitBlt(mem, 0, 0, w, h, screen, x, y, SRCCOPY | CAPTUREBLT);
        GdiFlush();
        f.width = w, f.height = h;
        f.pixels.assign(static_cast<uint8_t*>(bits), static_cast<uint8_t*>(bits) + size_t(w) * h * 4);
        SelectObject(mem, old);
    }
    if (dib) DeleteObject(dib);
    DeleteDC(mem);
    ReleaseDC(nullptr, screen);
    return f;
}

namespace detail {
struct ComInit {
    ComInit() { CoInitializeEx(nullptr, COINIT_MULTITHREADED); }
    ~ComInit() { CoUninitialize(); }
};
template <class T>
struct Com {
    T* p = nullptr;
    ~Com() { if (p) p->Release(); }
    T** operator&() { return &p; }
    T* operator->() { return p; }
};
inline IWICImagingFactory* factory() {
    static ComInit init;
    static IWICImagingFactory* f = [] {
        IWICImagingFactory* fac = nullptr;
        CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&fac));
        return fac;
    }();
    return f;
}
}  // namespace detail

inline Frame load_image(const std::string& utf8_path) {
    Frame f;
    IWICImagingFactory* fac = detail::factory();
    if (!fac) return f;
    detail::Com<IWICBitmapDecoder> dec;
    if (FAILED(fac->CreateDecoderFromFilename(widen(utf8_path).c_str(), nullptr, GENERIC_READ,
                                              WICDecodeMetadataCacheOnDemand, &dec)))
        return f;
    detail::Com<IWICBitmapFrameDecode> frame;
    if (FAILED(dec->GetFrame(0, &frame))) return f;
    detail::Com<IWICFormatConverter> conv;
    if (FAILED(fac->CreateFormatConverter(&conv))) return f;
    if (FAILED(conv->Initialize(frame.p, GUID_WICPixelFormat32bppBGRA, WICBitmapDitherTypeNone, nullptr, 0.0,
                                WICBitmapPaletteTypeCustom)))
        return f;
    UINT w = 0, h = 0;
    conv->GetSize(&w, &h);
    f.width = int32_t(w), f.height = int32_t(h);
    f.pixels.resize(size_t(w) * h * 4);
    if (FAILED(conv->CopyPixels(nullptr, w * 4, UINT(f.pixels.size()), f.pixels.data()))) f = Frame{};
    return f;
}

inline bool save_png(const std::string& utf8_path, const Frame& img) {
    IWICImagingFactory* fac = detail::factory();
    if (!fac || img.empty()) return false;
    detail::Com<IWICStream> stream;
    if (FAILED(fac->CreateStream(&stream))) return false;
    if (FAILED(stream->InitializeFromFilename(widen(utf8_path).c_str(), GENERIC_WRITE))) return false;
    detail::Com<IWICBitmapEncoder> enc;
    if (FAILED(fac->CreateEncoder(GUID_ContainerFormatPng, nullptr, &enc))) return false;
    if (FAILED(enc->Initialize(stream.p, WICBitmapEncoderNoCache))) return false;
    detail::Com<IWICBitmapFrameEncode> fr;
    if (FAILED(enc->CreateNewFrame(&fr, nullptr))) return false;
    fr->Initialize(nullptr);
    fr->SetSize(UINT(img.width), UINT(img.height));
    WICPixelFormatGUID fmt = GUID_WICPixelFormat32bppBGRA;
    fr->SetPixelFormat(&fmt);
    fr->WritePixels(UINT(img.height), UINT(img.width * 4), UINT(img.pixels.size()),
                    const_cast<BYTE*>(img.pixels.data()));
    fr->Commit();
    return SUCCEEDED(enc->Commit());
}

}  // namespace dks::win32
