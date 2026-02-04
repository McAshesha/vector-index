cmake_minimum_required(VERSION 3.16)

project(simsimd_jni C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(JNI REQUIRED)

include(FetchContent)

FetchContent_Declare(
    simsimd
    GIT_REPOSITORY https://github.com/ashvardanian/SimSIMD.git
    GIT_TAG main
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(simsimd)

add_library(simsimd_jni SHARED
    simsimd_jni.c
)

target_include_directories(simsimd_jni PRIVATE
    ${JNI_INCLUDE_DIRS}
    ${simsimd_SOURCE_DIR}/include
)

if (CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(simsimd_jni PRIVATE -O3 -march=native -ffast-math)
endif()

# target_link_libraries(simsimd_jni PRIVATE ${JNI_LIBRARIES}
