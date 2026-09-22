cmake_minimum_required(VERSION 3.20)
project(SAM3MaskOFX CXX)
option(SAM3_BUILD_TESTS "Build native host-cancellation regression checks" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_library(SAM3Mask SHARED src/sam3_ofx.cpp)
target_include_directories(SAM3Mask PRIVATE include/openfx)

if(WIN32)
  target_link_libraries(SAM3Mask PRIVATE ws2_32)
  set(OFX_ARCH_DIR "Win64")
  if(MSVC)
    target_compile_options(SAM3Mask PRIVATE /W3 /O2 /utf-8)
  endif()
else()
  set(OFX_ARCH_DIR "Linux-x86-64")
  target_compile_options(SAM3Mask PRIVATE -Wall -O2 -fvisibility=hidden -fPIC)
  target_link_libraries(SAM3Mask PRIVATE dl pthread)
  # keep every symbol but the OFX entry points private, and avoid depending
  # on a newer libstdc++ than the host ships
  target_link_options(SAM3Mask PRIVATE -Wl,--exclude-libs,ALL -static-libstdc++ -static-libgcc)
endif()

# OFX bundle layout: SAM3Mask.ofx.bundle/Contents/<arch>/SAM3Mask.ofx
set(BUNDLE_DIR "${CMAKE_BINARY_DIR}/SAM3Mask.ofx.bundle/Contents/${OFX_ARCH_DIR}")
set_target_properties(SAM3Mask PROPERTIES
  PREFIX ""
  SUFFIX ".ofx"
  RUNTIME_OUTPUT_DIRECTORY "${BUNDLE_DIR}"
  RUNTIME_OUTPUT_DIRECTORY_RELEASE "${BUNDLE_DIR}"
  RUNTIME_OUTPUT_DIRECTORY_DEBUG "${BUNDLE_DIR}"
  RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${BUNDLE_DIR}"
  LIBRARY_OUTPUT_DIRECTORY "${BUNDLE_DIR}")

if(SAM3_BUILD_TESTS)
  enable_testing()
  add_executable(sam3_native_cancel ../tests/native_cancel.cpp)
  target_include_directories(sam3_native_cancel PRIVATE include/openfx)
  if(WIN32)
    target_link_libraries(sam3_native_cancel PRIVATE ws2_32)
    target_compile_options(sam3_native_cancel PRIVATE /utf-8)
  else()
    target_link_libraries(sam3_native_cancel PRIVATE dl pthread)
  endif()
  add_test(NAME native_cancel COMMAND sam3_native_cancel)
endif()
