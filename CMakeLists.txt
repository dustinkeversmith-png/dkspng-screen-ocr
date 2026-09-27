cmake_minimum_required(VERSION 3.16)
project(dks_screen_ocr C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Header-only library.
add_library(dks INTERFACE)
target_include_directories(dks INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

if(MSVC)
  add_compile_options(/W4 /O2 /utf-8)
else()
  add_compile_options(-Wall -Wextra -O3 -march=native)
endif()

add_executable(test_core tests/test_core.cpp)
target_link_libraries(test_core PRIVATE dks)
add_executable(test_pipeline tests/test_pipeline.cpp)
target_link_libraries(test_pipeline PRIVATE dks)
enable_testing()
add_test(NAME test_core COMMAND test_core)
add_test(NAME test_pipeline COMMAND test_pipeline WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# Benchmarks / tools use the Win32 platform layer (WIC image IO, GDI capture).
if(WIN32)
  foreach(tool bench_segment bench_ocr ocr_debug screen_ocr bench_packs bench_latex screen_pipeline bench_latex_stage)
    add_executable(${tool} bench/${tool}.cpp)
    target_include_directories(${tool} PRIVATE bench)
    target_link_libraries(${tool} PRIVATE dks gdi32 ole32 windowscodecs uuid user32)
  endforeach()
endif()
