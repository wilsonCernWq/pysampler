cmake_minimum_required(VERSION 3.24)
project(pysampler VERSION 0.1.0 LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD    17)
set(CMAKE_CXX_EXTENSIONS  OFF)
set(CMAKE_CUDA_STANDARD   17)
set(ENABLE_CUDA           ON)

message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
find_package(CUDAToolkit REQUIRED)

# Detect _GLIBCXX_USE_CXX11_ABI from the active Python / PyTorch build.
# Users can override with -DGLIBCXX_USE_CXX11_ABI=0|1.
if(NOT DEFINED GLIBCXX_USE_CXX11_ABI)
  if(NOT Python_EXECUTABLE)
    find_package(Python COMPONENTS Interpreter QUIET)
  endif()
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import torch; print(torch.compiled_with_cxx11_abi())"
    OUTPUT_VARIABLE _torch_cxx11_abi
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
    RESULT_VARIABLE _torch_abi_rc
  )
  if(_torch_abi_rc EQUAL 0 AND _torch_cxx11_abi STREQUAL "True")
    set(GLIBCXX_USE_CXX11_ABI 1)
  else()
    set(GLIBCXX_USE_CXX11_ABI 0)
  endif()
endif()
message(STATUS "_GLIBCXX_USE_CXX11_ABI=${GLIBCXX_USE_CXX11_ABI}")
# Apply the ABI flag project-wide so every target — including all FetchContent
# subprojects (VTKm, OpenVKL, witcher, …) — uses a consistent ABI.
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=${GLIBCXX_USE_CXX11_ABI})

# Direct every shared library built in this project (extension, VTKm internals,
# OpenVKL, witcher, …) into one staging directory.  The install step then copies
# the whole directory as the pysampler package — no need to enumerate targets.
set(_pysampler_staging "${CMAKE_BINARY_DIR}/pysampler")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${_pysampler_staging}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${_pysampler_staging}")

# Allow the extension and co-installed libs to find each other via $ORIGIN.
# CMAKE_BUILD_WITH_INSTALL_RPATH makes the build-tree binaries already carry
# $ORIGIN as their RPATH, so install(DIRECTORY ...) can copy them verbatim
# without a separate CMake RPATH-patching step.
set(CMAKE_INSTALL_RPATH              "$ORIGIN")
set(CMAKE_BUILD_WITH_INSTALL_RPATH   ON)
set(CMAKE_POSITION_INDEPENDENT_CODE  ON)

# ── Dependencies ──────────────────────────────────────────────────────────────
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(FetchContent)
include(cmake/dep_pybind11.cmake)
include(cmake/dep_witcher.cmake)   # declares ENABLE_WITCHER; must precede the CUDA backend
include(cmake/dep_openvkl.cmake)   # declares ENABLE_OPENVKL; must precede the CUDA backend
include(cmake/dep_vtkm.cmake)      # declares ENABLE_VTKM; must precede the CUDA backend

# ── Generated headers ─────────────────────────────────────────────────────────
configure_file(config.h.in "${CMAKE_CURRENT_BINARY_DIR}/config.h")

set(_private_inc
  "${CMAKE_CURRENT_SOURCE_DIR}/csrc"
  "${CMAKE_CURRENT_BINARY_DIR}"
)

# ── sampler (core, static) ────────────────────────────────────────────────────
add_library(sampler STATIC
  "${CMAKE_CURRENT_SOURCE_DIR}/csrc/generate_random_numbers_serial.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/csrc/sampler.cpp"
)
target_include_directories(sampler
  PRIVATE ${_private_inc}
)
target_link_libraries(sampler
  PRIVATE pybind11::headers
)
set_target_properties(sampler PROPERTIES
  POSITION_INDEPENDENT_CODE ON
)

# ── Embed OptiX device code as a C array (ENABLE_WITCHER only) ────────────────
if(ENABLE_WITCHER)
  cuda_compile_and_embed(sampler_cuda_exa_devicecode
    "${CMAKE_CURRENT_SOURCE_DIR}/csrc/sampler_cuda_exa_devicecode.cu"
    LINK owl
    OPTIONS
      -I"${CMAKE_CURRENT_SOURCE_DIR}"
      -I"${CMAKE_CURRENT_BINARY_DIR}"
      -I"${witcher_SOURCE_DIR}"
      -DENABLE_WITCHER
  )
  set(_exa_c "${sampler_cuda_exa_devicecode}")
  target_sources(sampler PRIVATE "${_exa_c}")
endif()

# ── Required: sampler_cuda (CUDA backend, OBJECT) ───────────────────────────────────────
set(_cuda_sources
  "${CMAKE_CURRENT_SOURCE_DIR}/csrc/generate_random_numbers_cuda.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/csrc/sampler_cuda.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/csrc/sampler_cuda_regular.cu"
)
if(ENABLE_WITCHER)
  list(APPEND _cuda_sources "${CMAKE_CURRENT_SOURCE_DIR}/csrc/sampler_cuda_exa.cu")
endif()

add_library(sampler_cuda OBJECT ${_cuda_sources})
target_include_directories(sampler_cuda
  PRIVATE ${_private_inc} "${CMAKE_CURRENT_SOURCE_DIR}/csrc/cuda"
)
target_link_libraries(sampler_cuda
  PRIVATE CUDA::curand
)
target_compile_options(sampler_cuda
  PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda --expt-relaxed-constexpr>
)
set_target_properties(sampler_cuda PROPERTIES
  CUDA_STANDARD             17
  POSITION_INDEPENDENT_CODE ON
)
if(ENABLE_WITCHER)
  target_include_directories(sampler_cuda PRIVATE "${witcher_SOURCE_DIR}")
  target_link_libraries(sampler_cuda PRIVATE witcher_core owl)
endif()
# >>> link to sampler <<<
target_link_libraries(sampler PUBLIC sampler_cuda)

# ── Optional backends ─────────────────────────────────────────────────────────

if(ENABLE_OPENVKL)
  add_library(sampler_openvkl OBJECT 
    "${CMAKE_CURRENT_SOURCE_DIR}/csrc/sampler_openvkl.cpp"
  )
  target_include_directories(sampler_openvkl PRIVATE ${_private_inc})
  target_link_libraries(sampler_openvkl PRIVATE
    TBB::tbb openvkl::openvkl
    $<$<TARGET_EXISTS:openvkl::openvkl_module_cpu_device>:openvkl::openvkl_module_cpu_device>
    $<$<TARGET_EXISTS:OpenVDB::openvdb>:OpenVDB::openvdb>
  )
  target_compile_definitions(sampler_openvkl PRIVATE
    $<$<TARGET_EXISTS:OpenVDB::openvdb>:ENABLE_OPENVDB>
  )
  set_target_properties(sampler_openvkl PROPERTIES POSITION_INDEPENDENT_CODE ON)
  # >>> link to sampler <<<
  target_link_libraries(sampler PUBLIC sampler_openvkl)
endif()

if(ENABLE_VTKM)
  add_library(sampler_vtkm OBJECT 
    "${CMAKE_CURRENT_SOURCE_DIR}/csrc/sampler_vtkm.cxx"
  )
  target_include_directories(sampler_vtkm PRIVATE ${_private_inc})
  target_link_libraries(sampler_vtkm PRIVATE vtkm_cont vtkm_io)
  vtkm_add_target_information(sampler_vtkm
    DROP_UNUSED_SYMBOLS
    MODIFY_CUDA_FLAGS
    DEVICE_SOURCES
      "${CMAKE_CURRENT_SOURCE_DIR}/csrc/sampler_vtkm.cxx"
  )
  set_target_properties(sampler_vtkm PROPERTIES POSITION_INDEPENDENT_CODE ON)
  # >>> link to sampler <<<
  target_link_libraries(sampler PUBLIC sampler_vtkm)
endif()

# ── pysampler (pybind11 extension) ────────────────────────────────────────────
set(ext_name pysampler)
pybind11_add_module(${ext_name} "${CMAKE_CURRENT_SOURCE_DIR}/csrc/pysampler.cpp")
target_include_directories(${ext_name}
  PRIVATE ${_private_inc}
)
target_link_libraries(${ext_name}
  PRIVATE
    pybind11::pybind11
    pybind11::module
    pybind11::lto
    pybind11::windows_extras
    sampler
)
target_compile_definitions(${ext_name} PRIVATE EXTENSION_NAME=${ext_name})
pybind11_extension(${ext_name})
if(NOT MSVC AND NOT "${CMAKE_BUILD_TYPE}" MATCHES "Debug|RelWithDebInfo")
  pybind11_strip(${ext_name})
endif()
set_target_properties(${ext_name} PROPERTIES
  CXX_VISIBILITY_PRESET  "hidden"
  CUDA_VISIBILITY_PRESET "hidden"
)

# ── Install ───────────────────────────────────────────────────────────────────
# install(TARGETS ...) is used for every dep so CMake can find each target's
# actual output regardless of where the FetchContent subproject built it, AND
# so CMake patches the RPATH to $ORIGIN at install time.
#
# VTKm and OpenVKL build to their own _deps/*/build/ subdirectories and do NOT
# respect CMAKE_LIBRARY_OUTPUT_DIRECTORY, so a directory-glob install would miss
# them.  install(TARGETS) is the only reliable approach.

install(FILES
  "${CMAKE_CURRENT_SOURCE_DIR}/python/__init__.py"
  DESTINATION pysampler
  COMPONENT pysampler
)

# pybind11 extension (MODULE library → LIBRARY destination on Linux)
set(_pysampler_targets ${ext_name})

if(ENABLE_WITCHER)
  # umesh may land in the build root (not the staging dir); install(TARGETS)
  # finds it regardless.
  list(APPEND _pysampler_targets witcher_core owl umesh)
endif()

if(ENABLE_VTKM)
  # vtkmdiympi_nompi is the actual shared lib; vtkmdiy is INTERFACE-only.
  list(APPEND _pysampler_targets vtkm_cont vtkm_io vtkmdiympi_nompi)
endif()

if(ENABLE_OPENVKL)
  list(APPEND _pysampler_targets openvkl openvkl_module_cpu_device rkcommon embree)
  list(APPEND _pysampler_targets openvkl_module_cpu_device_4 openvkl_module_cpu_device_8 openvkl_module_cpu_device_16)
endif()

# Install all targets in the _pysampler_targets list
foreach(_t ${_pysampler_targets})
  if(TARGET ${_t})
    install(TARGETS ${_t}
      LIBRARY DESTINATION pysampler COMPONENT pysampler
      RUNTIME DESTINATION pysampler COMPONENT pysampler
    )
  endif()
endforeach()
