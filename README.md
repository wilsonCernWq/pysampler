# pysampler

CUDA volume sampler — zero-copy access to structured, OpenVDB, and AMR volumes.

`pysampler` is built to accelerate the training of **Volumetric Neural
Representations (VNRs)** — networks that encode a volume as a continuous
function $f \colon (x, y, z) \mapsto v$. Training such networks requires drawing
millions of `(coord, value)` pairs per second from the ground-truth volume,
and `pysampler` is the data-loading half of that loop: it fuses random
coordinate generation and volume sampling into a single CUDA kernel that
writes directly into PyTorch tensors, eliminating host-device transfers in
the hot path.

It is the sampling backend used by `instantvnr`:

- **Project:** [VIDILabs/instantvnr](https://github.com/VIDILabs/instantvnr)
  — interactive volumetric-neural-representation training & rendering.
- **Paper:** Wu et al., *Interactive Volume Visualization via Multi-Resolution
  Hash Encoding Based Neural Representation*, IEEE TVCG 2024 &nbsp;·&nbsp;
  [DOI](https://doi.org/10.1109/TVCG.2023.3293121) &nbsp;·&nbsp;
  [arXiv:2207.11620](https://arxiv.org/abs/2207.11620)

See [Citation](#citation) below for the BibTeX entry.

## Overview

`pysampler` is a [pybind11](https://pybind11.readthedocs.io/) extension that
exposes a single `Sampler` object backed by several volume-sampling
implementations (CUDA `tex3D`, OpenVKL, OWL/OptiX). All sampling routines
write results directly into caller-owned CUDA device buffers via raw pointers,
which makes the module trivially interoperable with PyTorch tensors (`tensor.data_ptr()`)
and other CUDA-aware libraries — no host round-trip, no extra copies.

The Python surface is intentionally tiny:

- `pysampler.create_sampler(type, device, **kwargs)` — build a sampler for a
  given volume type and execution device.
- `pysampler.sample(sampler, coords_ptr, values_ptr, count)` — generate random
  coordinates *and* sample, both written to caller buffers.
- `pysampler.decode(sampler, coords_ptr, values_ptr, count)` — sample at
  caller-supplied coordinates.
- `Sampler.n_channels()` — number of scalar channels in the volume.

The full C++ binding is in `csrc/pysampler.cpp`.

## Features

- **Structured grids** on CUDA (`tex3D` hardware trilinear) and on CPU via
  OpenVKL.
- **OpenVDB** sparse grids via OpenVKL (CPU).
- **VTK-m** structured-mesh I/O (`.vtk`, `.vti`, `.pvti`, …) feeding the OpenVKL
  device.
- **AMR** — ExaBrick and ExaStitch — via the OWL/OptiX-based `owlExaStitcher`
  ("witcher") backend.
- **Zero-copy** sampling — values and coordinate buffers are addressed by raw
  `uintptr_t` device pointers; ownership stays with the Python caller.
- Build via `scikit-build-core` + CMake, C++17, CUDA, with all heavy
  dependencies (OpenVKL/Embree/rkcommon, VTK-m, owlExaStitcher) pulled in via
  `FetchContent` and staged into the wheel.

## Requirements

- NVIDIA GPU + CUDA Toolkit (the CUDA backend is mandatory; CPU-only builds
  are not supported).
- Python ≥ 3.9.
- [`uv`](https://docs.astral.sh/uv/) — used by `setup_venv.sh`.
- ISPC ≥ 1.30 — auto-installed into the venv by `setup_venv.sh` (required by
  the OpenVKL CPU device).
- TBB headers + libs (`libtbb-dev` or equivalent) — required when
  `ENABLE_OPENVKL=ON`.
- NVIDIA OptiX SDK — required when `ENABLE_WITCHER=ON`; located via
  `cmake/FindOptiX.cmake` (see `cmake/configure_optix.cmake`).
- PyTorch — automatically installed; `setup_venv.sh` selects the wheel index
  matching the local CUDA toolkit (and switches to `nightly/cu128` for
  Blackwell, `sm_120+`).

## Installation

### Recommended: `setup_venv.sh`

From the `pysampler/` directory:

```bash
./setup_venv.sh
```

The script:

1. Detects GPU compute capability (`nvidia-smi --query-gpu=compute_cap`) and the
   CUDA toolkit (`nvcc`).
2. Picks the matching PyTorch wheel index (`cu118` / `cu121` / `cu124` / `cu128`,
   or `nightly/cu128` for Blackwell).
3. Creates a `uv` venv (default `.venv`, Python 3.11) and installs ISPC 1.30
   into `<venv>/bin`.
4. Runs `uv pip install -v .[test]`, which builds the C++ extension via
   `scikit-build-core` and installs `pysampler` plus test dependencies.

Useful overrides:

```bash
SM=86 ./setup_venv.sh                # force compute capability
VENV_DIR=~/envs/pysampler ./setup_venv.sh
./setup_venv.sh --clean              # remove .venv and clear uv cache
```

### Manual

```bash
uv venv .venv --python 3.11
source .venv/bin/activate
# Make sure `ispc` and `nvcc` are on PATH.
uv pip install .[test]
```

To toggle backends, pass cache variables through `scikit-build-core`:

```bash
uv pip install -v . \
  --config-settings cmake.define.ENABLE_OPENVKL=OFF \
  --config-settings cmake.define.ENABLE_WITCHER=OFF \
  --config-settings cmake.define.ENABLE_VTKM=OFF \
  --config-settings cmake.define.CMAKE_CUDA_ARCHITECTURES=86
```

Defaults from `pyproject.toml`: `ENABLE_OPENVKL`, `ENABLE_WITCHER`, and
`ENABLE_VTKM` are all `ON`; `CMAKE_CUDA_ARCHITECTURES=native`.

## Quick start

A minimal end-to-end example: load a `.raw` volume on CUDA and sample it at
PyTorch-managed coordinates. Adapted from `tests/test_sampler.py`.

```python
import numpy as np
import torch
import pysampler

Dx, Dy, Dz = 32, 40, 48
volume = np.random.default_rng(0).standard_normal((Dz, Dy, Dx)).astype(np.float32)
volume.tofile("volume.raw")

sampler = pysampler.create_sampler(
    "structuredRegular", "cuda",
    filename="volume.raw",
    dims=[Dx, Dy, Dz],
    dtype="float32",
)

N = 4096
coords = torch.rand((N, 3), dtype=torch.float32, device="cuda")           # (N, 3)
values = torch.empty((sampler.n_channels(), N), dtype=torch.float32, device="cuda")  # (C, N)

pysampler.decode(sampler, coords.data_ptr(), values.data_ptr(), N)

# values[c, i] is the c-th channel sampled at coords[i]
```

## Per-backend usage

The `device × volume type` support matrix mirrors the dispatch in
`csrc/sampler.cpp`:

| volume type         | `cuda`              | `openvkl` |
| ------------------- | ------------------- | --------- |
| `structuredRegular` | yes                 | yes       |
| `openvdb`           | —                   | yes       |
| `vtkm`              | —                   | yes       |
| `exabrick`          | yes (`ENABLE_WITCHER`) | —      |
| `exastitch`         | yes (`ENABLE_WITCHER`) | —      |

### `structuredRegular`

Required: `dims=[Dx, Dy, Dz]`, `dtype` (one of `uint8`, `int8`, `uint16`,
`int16`, `uint32`, `int32`, `float`/`float32`, `double`/`float64`).
Optional: `spacing=[sx, sy, sz]` (default `[1, 1, 1]`), `n_channels` (default
`1`), `filename`, `offset` (byte offset into file, default `0`),
`is_big_endian` (default `False`).

```python
sampler = pysampler.create_sampler(
    "structuredRegular", "cuda",
    filename="volume.raw",
    dims=[256, 256, 256],
    dtype="uint16",
    spacing=[1.0, 1.0, 2.0],
    offset=0,
    is_big_endian=False,
)
```

### `openvdb`

Requires `ENABLE_OPENVKL=ON`.

```python
sampler = pysampler.create_sampler(
    "openvdb", "openvkl",
    filename="bunny.vdb",
    field="density",
)
```

### `vtkm`

Requires `ENABLE_OPENVKL=ON` and `ENABLE_VTKM=ON`.

```python
sampler = pysampler.create_sampler(
    "vtkm", "openvkl",
    files=["timestep_0.vti", "timestep_1.vti"],
    field="temperature",
)
```

### `exabrick`

Requires `ENABLE_WITCHER=ON` (OptiX).

```python
sampler = pysampler.create_sampler(
    "exabrick", "cuda",
    bricks="dataset.bricks",
    scalar="dataset.scalar",
)
```

### `exastitch`

Requires `ENABLE_WITCHER=ON` (OptiX).

```python
sampler = pysampler.create_sampler(
    "exastitch", "cuda",
    umesh="dataset.umesh",
    grids="dataset.grids",
    scalar="dataset.scalar",
)
```

## Zero-copy / PyTorch interop

Every sampling call takes raw device pointers and a count; the Python caller
allocates and owns the buffers. The contract (from `csrc/pysampler.cpp`):

- `coords_ptr` — `float32` device buffer, shape `(count, 3)`. Holds sample
  positions in `[0, 1]^3`.
- `values_ptr` — `float32` device buffer, shape `(n_channels, count)`,
  column-major over samples. Transpose on the Python side if you want a
  `(count, n_channels)` view.

PyTorch tensors expose the right pointer via `tensor.data_ptr()`:

```python
coords = torch.empty((N, 3),                       dtype=torch.float32, device="cuda")
values = torch.empty((sampler.n_channels(), N),    dtype=torch.float32, device="cuda")

# decode(): caller supplies coordinates
pysampler.decode(sampler, coords.data_ptr(), values.data_ptr(), N)

# sample(): sampler fills BOTH buffers — coords get random uniforms in [0,1]^3
pysampler.sample(sampler, coords.data_ptr(), values.data_ptr(), N)
```

Use `decode()` whenever you already know the coordinates (e.g. an INR query
batch). Use `sample()` to draw fresh uniform samples in one fused kernel
launch — `coords` is overwritten with the random positions used.

## Build options

CMake cache variables exposed via `pyproject.toml`'s `[tool.scikit-build.cmake.define]`:

| Variable                  | Default  | Effect |
| ------------------------- | -------- | ------ |
| `CMAKE_CUDA_ARCHITECTURES` | `native` | CUDA architecture(s) to target. |
| `ENABLE_OPENVKL`          | `ON`     | Pull in rkcommon + Embree + OpenVKL (CPU sampling for structured / OpenVDB / VTK-m). |
| `ENABLE_WITCHER`          | `ON`     | Pull in `owlExaStitcher` for the ExaBrick / ExaStitch backends (requires OptiX). |
| `ENABLE_VTKM`             | `ON`     | Build VTK-m 2.3.0 from source for structured-mesh I/O. |

`_GLIBCXX_USE_CXX11_ABI` is auto-detected from the active PyTorch (see
`CMakeLists.txt`) and applied project-wide so all FetchContent subprojects
agree on the ABI. Override with `-DGLIBCXX_USE_CXX11_ABI=0|1` if needed.

All shared libraries built by the project (the extension itself plus
co-installed VTK-m, OpenVKL, owl, umesh, …) are staged into a single
`pysampler/` directory inside the wheel and use `$ORIGIN` RPATH so they
locate each other at runtime.

## Testing

Tests are CUDA-only and are auto-skipped when no GPU is available
(see `pytestmark` in `tests/test_sampler.py`).

```bash
source .venv/bin/activate
pytest -v tests
```

The reference test (`test_decode_trilinear_matches_reference`) compares the
CUDA `tex3D` output against a software trilinear implementation with
`atol=3e-3`, accounting for the 9-bit interpolation-weight precision used
by NVIDIA texture units.

## Project layout

```
pysampler/
├── CMakeLists.txt          # top-level build: targets, install, RPATH
├── pyproject.toml          # scikit-build-core configuration
├── setup_venv.sh           # one-shot venv + build helper (uv + ISPC + PyTorch)
├── config.h.in             # ENABLE_* macros consumed by C++ sources
├── conftest.py             # keeps pytest from importing the source tree
├── cmake/                  # FetchContent recipes (witcher, openvkl, vtkm, …)
├── csrc/                   # C++/CUDA sources
│   ├── pysampler.cpp       # pybind11 module entry point
│   ├── sampler.{h,cpp}     # base Sampler + create_sampler dispatch
│   ├── sampler_cuda*.cu    # CUDA backend (tex3D structured + Exa AMR)
│   ├── sampler_openvkl.*   # OpenVKL backend (CPU)
│   └── sampler_vtkm.*      # VTK-m loader feeding OpenVKL
├── python/__init__.py      # `from pysampler.pysampler import *`
└── tests/test_sampler.py   # reference trilinear + smoke tests
```

## Citation

If you use `pysampler` in academic work, please cite this repository:

```bibtex
@software{pysampler,
  author  = {Wu, Qi},
  title   = {{pysampler}: {CUDA} volume sampler for {V}olumetric {N}eural {R}epresentations},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/wilsonCernWq/pysampler},
}
```

You may also want to cite the underlying volumetric neural representation
paper that motivates this sampler:

- Qi Wu, David Bauer, Michael J. Doyle, Kwan-Liu Ma. *Interactive Volume
  Visualization via Multi-Resolution Hash Encoding Based Neural
  Representation*. IEEE Transactions on Visualization and Computer Graphics,
  vol. 30, 2024.
  [DOI: 10.1109/TVCG.2023.3293121](https://doi.org/10.1109/TVCG.2023.3293121)
  ·
  [arXiv:2207.11620](https://arxiv.org/abs/2207.11620)
  ·
  [code](https://github.com/VIDILabs/instantvnr)

## License

Apache-2.0 — see [LICENSE](LICENSE). Copyright 2026 Qi Wu.

## Author

Qi Wu — `wilson.over.cloud@gmail.com`.
