# ===========================================================================
# CMakeLists.txt -- Cross-platform build for the SimSIMD JNI shared library
# ===========================================================================
#
# Produces: native/<os>-<arch>/<libname>
#   Linux:   native/linux-amd64/libsimsimd_jni.so
#   macOS:   native/macos-aarch64/libsimsimd_jni.dylib
#   Windows: native/windows-amd64/simsimd_jni.dll
#
# Build (handled automatically by Maven):
#   cmake -S src/main/native -B target/native-build -DCMAKE_BUILD_TYPE=Release
#   cmake --build target/native-build --config Release
# ===========================================================================

cmake_minimum_required(VERSION 3.16)
project(simsimd_jni C)

# ---------------------------------------------------------------------------
# Compiler standards
# ---------------------------------------------------------------------------
set(CMAKE_C_STANDARD 99)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ---------------------------------------------------------------------------
# Platform detection -- produces NATIVE_OS and NATIVE_ARCH matching
# the naming convention used by NativeLibLoader.java
# ---------------------------------------------------------------------------
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(NATIVE_OS "linux")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    set(NATIVE_OS "macos")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(NATIVE_OS "windows")
else()
    message(FATAL_ERROR "Unsupported OS: ${CMAKE_SYSTEM_NAME}")
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
    set(NATIVE_ARCH "amd64")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
    set(NATIVE_ARCH "aarch64")
else()
    message(FATAL_ERROR "Unsupported architecture: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

message(STATUS "Target platform: ${NATIVE_OS}-${NATIVE_ARCH}")

# ---------------------------------------------------------------------------
# Output directories -- native/<os>-<arch>/ under the build tree.
# CMAKE_RUNTIME_OUTPUT_DIRECTORY is needed for Windows DLLs.
# ---------------------------------------------------------------------------
set(NATIVE_OUTPUT_DIR ${CMAKE_BINARY_DIR}/native/${NATIVE_OS}-${NATIVE_ARCH})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${NATIVE_OUTPUT_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${NATIVE_OUTPUT_DIR})

# Multi-config generators (Visual Studio, Xcode) append a per-config subdir
# (e.g. Release/) unless we suppress it for every config.
foreach(CONFIG_TYPE Release Debug RelWithDebInfo MinSizeRel)
    string(TOUPPER ${CONFIG_TYPE} CONFIG_UPPER)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${CONFIG_UPPER} ${NATIVE_OUTPUT_DIR})
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${CONFIG_UPPER} ${NATIVE_OUTPUT_DIR})
endforeach()

# ---------------------------------------------------------------------------
# Locate JNI headers (jni.h, jni_md.h)
# ---------------------------------------------------------------------------
find_package(JNI REQUIRED)

# ---------------------------------------------------------------------------
# Fetch SimSIMD (header-only SIMD distance library)
# ---------------------------------------------------------------------------
# TODO: pin to a release tag (e.g. GIT_TAG v6.2.1) for reproducible builds
include(FetchContent)
FetchContent_Declare(
    simsimd
    GIT_REPOSITORY https://github.com/ashvardanian/SimSIMD.git
    GIT_TAG        main
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(simsimd)

# ---------------------------------------------------------------------------
# Shared library target
# ---------------------------------------------------------------------------
add_library(simsimd_jni SHARED simsimd_jni.c)

target_include_directories(simsimd_jni PRIVATE
    ${JNI_INCLUDE_DIRS}
    ${simsimd_SOURCE_DIR}/include
)

# ---------------------------------------------------------------------------
# Platform-specific optimization flags
# ---------------------------------------------------------------------------
# -march=native generates code for the host CPU (AVX-512, AVX2, NEON, etc.).
# The resulting binary is optimal for the build machine but not portable
# to CPUs with fewer instruction sets.  This is acceptable for a library
# that users build from source.
if(MSVC)
    target_compile_options(simsimd_jni PRIVATE /O2 /fp:fast)
    if(NATIVE_ARCH STREQUAL "amd64")
        target_compile_options(simsimd_jni PRIVATE /arch:AVX2)
    endif()
elseif(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(simsimd_jni PRIVATE -O3 -march=native -ffast-math)
endif()

# ---------------------------------------------------------------------------
# Windows: suppress the "lib" prefix so the file is simsimd_jni.dll,
# matching System.mapLibraryName("simsimd_jni") on Windows.
# On Unix, CMake already produces lib<name>.so / lib<name>.dylib by default,
# which matches System.mapLibraryName("simsimd_jni").
# ---------------------------------------------------------------------------
if(WIN32)
    set_target_properties(simsimd_jni PROPERTIES PREFIX "")
endif()
