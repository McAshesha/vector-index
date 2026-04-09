# ===========================================================================
# CMakeLists.txt -- Build configuration for the SimSIMD JNI shared library
# ===========================================================================
#
# This CMake project compiles the JNI bridge (simsimd_jni.c) into a native
# shared library (libsimsimd_jni.so on Linux, libsimsimd_jni.dylib on macOS).
#
# The resulting library is loaded at runtime by Java's System.loadLibrary()
# call in the SimSIMD class, providing hardware-accelerated vector distance
# computations (L2, dot product, cosine, Hamming) via the SimSIMD C library.
#
# Build invocation (from the project root, handled automatically by Maven):
#   cmake -S src/main/native -B src/main/native/build
#   cmake --build src/main/native/build
# ===========================================================================

# ---------------------------------------------------------------------------
# Minimum CMake version and project declaration
# ---------------------------------------------------------------------------
# CMake 3.16+ is required for FetchContent_MakeAvailable (introduced in 3.14)
# and modern JNI discovery.  The project is pure C (no C++ needed).
cmake_minimum_required(VERSION 3.16)

project(simsimd_jni C)

# ---------------------------------------------------------------------------
# Compiler standards and shared-library settings
# ---------------------------------------------------------------------------
# C99 is sufficient for both the JNI glue code and the SimSIMD headers.
# Position-independent code (-fPIC) is mandatory for objects that will be
# linked into a shared library; setting it globally ensures all translation
# units (including fetched dependencies) are compiled with -fPIC.
set(CMAKE_C_STANDARD 99)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ---------------------------------------------------------------------------
# Locate JNI headers (jni.h, jni_md.h)
# ---------------------------------------------------------------------------
# find_package(JNI REQUIRED) detects the active JDK via JAVA_HOME or system
# paths and populates JNI_INCLUDE_DIRS with the directories containing
# jni.h and the platform-specific jni_md.h.
find_package(JNI REQUIRED)

# ---------------------------------------------------------------------------
# Fetch SimSIMD from GitHub (header-only SIMD distance library)
# ---------------------------------------------------------------------------
# FetchContent downloads the SimSIMD repository at configure time so that its
# headers are available without requiring a manual install.  GIT_SHALLOW TRUE
# clones only the latest commit on the main branch to minimize download size.
include(FetchContent)

FetchContent_Declare(
    simsimd
    GIT_REPOSITORY https://github.com/ashvardanian/SimSIMD.git
    GIT_TAG main
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(simsimd)

# ---------------------------------------------------------------------------
# Define the shared library target
# ---------------------------------------------------------------------------
# The single source file (simsimd_jni.c) implements four JNI native methods.
# Building as SHARED produces a dynamically loadable library that the JVM can
# load via System.loadLibrary("simsimd_jni").
add_library(simsimd_jni SHARED
    simsimd_jni.c
)

# ---------------------------------------------------------------------------
# Include directories
# ---------------------------------------------------------------------------
# Two sets of headers are needed:
#   1. JNI_INCLUDE_DIRS  -- jni.h and jni_md.h from the JDK
#   2. simsimd include   -- simsimd/simsimd.h from the fetched source tree
# Both are PRIVATE because they are only needed to compile this library, not
# by consumers that link against it.
target_include_directories(simsimd_jni PRIVATE
    ${JNI_INCLUDE_DIRS}
    ${simsimd_SOURCE_DIR}/include
)

# ---------------------------------------------------------------------------
# Aggressive optimization flags (GCC and Clang only)
# ---------------------------------------------------------------------------
#   -O3           : highest standard optimization level; enables vectorization,
#                   inlining, loop unrolling, and other transformations.
#   -march=native : generate instructions for the host CPU's exact
#                   microarchitecture (AVX-512, AVX2, SSE4.2, NEON, etc.).
#                   This maximizes SIMD throughput but means the binary is
#                   NOT portable to older CPUs.
#   -ffast-math   : relax IEEE 754 compliance for faster floating-point code
#                   (allows reordering, fused multiply-add, etc.).  Acceptable
#                   here because approximate distance comparisons tolerate
#                   small numerical differences.
if (CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(simsimd_jni PRIVATE -O3 -march=native -ffast-math)
endif()

# ---------------------------------------------------------------------------
# Linking notes
# ---------------------------------------------------------------------------
# On most platforms the JNI shared library does NOT need to link against
# libjvm; the JVM provides the JNI function table at load time.  The
# commented-out line below is retained for reference in case explicit linking
# is ever required on a specific platform.
# target_link_libraries(simsimd_jni PRIVATE ${JNI_LIBRARIES})
