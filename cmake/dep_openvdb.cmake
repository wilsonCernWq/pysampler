cmake_minimum_required(VERSION 3.24)
include(FetchContent)

option(ENABLE_OPENVDB "Enable OpenVDB .vdb file loading (via OpenVKL bridge)" OFF)
if(NOT ENABLE_OPENVDB)
  return()
endif()

if(NOT ENABLE_OPENVKL)
  message(FATAL_ERROR
    "ENABLE_OPENVDB requires ENABLE_OPENVKL: .vdb files are sampled through "
    "OpenVKL's vdb volume type, which is only built when ENABLE_OPENVKL=ON.")
endif()

# ── 1. Prefer a system install ────────────────────────────────────────────────
# A system-installed OpenVDB matches the system TBB / Blosc / Boost ABI and
# avoids a multi-minute subproject build.  If found, we just use it.
find_package(OpenVDB CONFIG QUIET)
if(TARGET OpenVDB::openvdb)
  message(STATUS "Found system OpenVDB ${OpenVDB_VERSION}; skipping FetchContent build.")
  return()
endif()

# ── 2. Otherwise, build from source via FetchContent ──────────────────────────
#
# OpenVDB has three hard runtime dependencies (TBB, ZLIB, plus Blosc when most
# real-world .vdb files are involved).  We expect TBB and ZLIB from the system
# (libtbb-dev, zlib1g-dev) — both are standard packages on every supported
# distro.  Blosc is optional but strongly recommended because virtually every
# `.vdb` file produced by Houdini / Blender / Maya is Blosc-compressed.

find_package(TBB CONFIG QUIET)
if(NOT TARGET TBB::tbb)
  message(FATAL_ERROR "ENABLE_OPENVDB requires TBB. Install libtbb-dev or set TBB_DIR.")
endif()

find_package(ZLIB QUIET)
if(NOT TARGET ZLIB::ZLIB)
  message(FATAL_ERROR "ENABLE_OPENVDB requires zlib. Install zlib1g-dev.")
endif()

# Blosc has no Config package on most distros — probe for the library directly.
find_library(_openvdb_blosc_lib NAMES blosc)
if(_openvdb_blosc_lib)
  set(_openvdb_use_blosc ON)
  message(STATUS "OpenVDB: enabling Blosc support (${_openvdb_blosc_lib})")
else()
  set(_openvdb_use_blosc OFF)
  message(WARNING
    "libblosc not found; OpenVDB will be built without Blosc compression. "
    "Most real-world .vdb files will fail to load — install libblosc-dev.")
endif()
unset(_openvdb_blosc_lib CACHE)

set(OPENVDB_VERSION "12.0.0" CACHE STRING "OpenVDB version")

# ── Trim the OpenVDB build to just the shared core library ────────────────────
# Order in this block matches OpenVDB's CMakeLists.txt to make diffs easy.
set(OPENVDB_BUILD_CORE                      ON                    CACHE BOOL "" FORCE)
set(OPENVDB_BUILD_BINARIES                  OFF                   CACHE BOOL "" FORCE)
set(OPENVDB_BUILD_DOCS                      OFF                   CACHE BOOL "" FORCE)
set(OPENVDB_BUILD_UNITTESTS                 OFF                   CACHE BOOL "" FORCE)
set(OPENVDB_BUILD_PYTHON_MODULE             OFF                   CACHE BOOL "" FORCE)
set(OPENVDB_BUILD_AX                        OFF                   CACHE BOOL "" FORCE)
set(OPENVDB_BUILD_NANOVDB                   OFF                   CACHE BOOL "" FORCE)
set(OPENVDB_BUILD_VDB_PRINT                 OFF                   CACHE BOOL "" FORCE)
set(OPENVDB_BUILD_VDB_LOD                   OFF                   CACHE BOOL "" FORCE)
set(OPENVDB_BUILD_VDB_RENDER                OFF                   CACHE BOOL "" FORCE)
set(OPENVDB_BUILD_VDB_VIEW                  OFF                   CACHE BOOL "" FORCE)
set(OPENVDB_BUILD_VDB_TOOL                  OFF                   CACHE BOOL "" FORCE)
set(OPENVDB_CORE_SHARED                     ON                    CACHE BOOL "" FORCE)
set(OPENVDB_CORE_STATIC                     OFF                   CACHE BOOL "" FORCE)
# Delayed loading is the only OpenVDB-12 feature that still pulls in Boost.
# We don't need it (sampler_openvkl.cpp eagerly reads the whole grid), so turn
# it off and skip the Boost dependency entirely.
set(OPENVDB_USE_DELAYED_LOADING             OFF                   CACHE BOOL "" FORCE)
set(OPENVDB_DISABLE_BOOST_IMPLICIT_LINKING  ON                    CACHE BOOL "" FORCE)
# Don't define a top-level `uninstall` custom target — Embree (also FetchContent'd
# by dep_openvkl.cmake) defines one with the same name, which CMake refuses
# under policy CMP0002.
set(OPENVDB_ENABLE_UNINSTALL                OFF                   CACHE BOOL "" FORCE)
# CONCURRENT_MALLOC only links into OpenVDB binaries / unit tests, both of
# which are off above.  Setting it to None skips the find_package(Jemalloc)
# probe and silences the "Unable to find Jemalloc" warning at configure time.
set(CONCURRENT_MALLOC                       "None"                CACHE STRING "" FORCE)
# Don't generate explicit instantiations — they slow the OpenVDB build a lot
# and we only ever instantiate FloatGrid / Vec3SGrid in sampler_openvkl.cpp.
set(USE_EXPLICIT_INSTANTIATION              OFF                   CACHE BOOL "" FORCE)
# Optional features we skip: half-float grids (USE_IMATH_HALF) and pkg-config
# file generation (we install via CMake, not pkg-config).
set(USE_IMATH_HALF                          OFF                   CACHE BOOL "" FORCE)
set(USE_PKGCONFIG                           OFF                   CACHE BOOL "" FORCE)
set(USE_TBB                                 ON                    CACHE BOOL "" FORCE)
set(USE_ZLIB                                ON                    CACHE BOOL "" FORCE)
set(USE_BLOSC                               ${_openvdb_use_blosc} CACHE BOOL "" FORCE)

FetchContent_Declare(openvdb
  GIT_REPOSITORY  https://github.com/AcademySoftwareFoundation/openvdb.git
  GIT_TAG         v${OPENVDB_VERSION}
  GIT_SHALLOW     ON
  OVERRIDE_FIND_PACKAGE
)
FetchContent_MakeAvailable(openvdb)

# When built as a subproject, OpenVDB exports `openvdb_shared` (and/or
# `openvdb_static`) but does NOT create the `OpenVDB::openvdb` IMPORTED alias
# that find_package(OpenVDB) would.  Add the alias here so the rest of the
# build (and the OpenVKL bridge in sampler_openvkl.cpp) can link to it
# uniformly regardless of how OpenVDB was acquired.
if(NOT TARGET OpenVDB::openvdb)
  if(TARGET openvdb_shared)
    add_library(OpenVDB::openvdb ALIAS openvdb_shared)
  elseif(TARGET openvdb)
    add_library(OpenVDB::openvdb ALIAS openvdb)
  else()
    message(FATAL_ERROR
      "OpenVDB FetchContent succeeded but no openvdb / openvdb_shared target "
      "was created — this likely means OPENVDB_BUILD_CORE was overridden.")
  endif()
endif()
