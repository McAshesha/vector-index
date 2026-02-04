# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/home/ashesha/Sources/vector-index-bench/src/main/native/build/_deps/simsimd-src")
  file(MAKE_DIRECTORY "/home/ashesha/Sources/vector-index-bench/src/main/native/build/_deps/simsimd-src")
endif()
file(MAKE_DIRECTORY
  "/home/ashesha/Sources/vector-index-bench/src/main/native/build/_deps/simsimd-build"
  "/home/ashesha/Sources/vector-index-bench/src/main/native/build/_deps/simsimd-subbuild/simsimd-populate-prefix"
  "/home/ashesha/Sources/vector-index-bench/src/main/native/build/_deps/simsimd-subbuild/simsimd-populate-prefix/tmp"
  "/home/ashesha/Sources/vector-index-bench/src/main/native/build/_deps/simsimd-subbuild/simsimd-populate-prefix/src/simsimd-populate-stamp"
  "/home/ashesha/Sources/vector-index-bench/src/main/native/build/_deps/simsimd-subbuild/simsimd-populate-prefix/src"
  "/home/ashesha/Sources/vector-index-bench/src/main/native/build/_deps/simsimd-subbuild/simsimd-populate-prefix/src/simsimd-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/ashesha/Sources/vector-index-bench/src/main/native/build/_deps/simsimd-subbuild/simsimd-populate-prefix/src/simsimd-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/ashesha/Sources/vector-index-bench/src/main/native/build/_deps/simsimd-subbuild/simsimd-populate-prefix/src/simsimd-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
