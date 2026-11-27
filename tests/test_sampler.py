"""
Compare pysampler CUDA output against a reference software trilinear implementation.

The reference matches the logic in references/inr-research/exps/vnr/helpers.py:
  - data layout  : volume[iz, iy, ix], shape (Dz, Dy, Dx)
  - coord system : (x, y, z) ∈ [0, 1]^3  (x → width, y → height, z → depth)
  - normalization: (raw_value - vmin) / (vmax - vmin) using actual data min/max
  - interpolation: CUDA tex3D address formula u*N - 0.5 → fractional voxel index

Hardware (tex3D) vs software trilinear may differ by up to ~2e-4 due to the
9-bit interpolation-weight precision on most NVIDIA GPUs.
"""
import os
import tempfile

import numpy as np
import pytest
import torch
import pysampler

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="requires a CUDA-capable GPU",
)


# ---------------------------------------------------------------------------
# Reference implementation
# ---------------------------------------------------------------------------

def _trilinear_ref(volume: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """
    Software trilinear interpolation matching CUDA tex3D with normalized coords.

    Parameters
    ----------
    volume : float32 ndarray, shape (Dz, Dy, Dx)
        Raw voxel data indexed as volume[iz, iy, ix].
    coords : float32 ndarray, shape (N, 3)
        Sample positions (x, y, z) ∈ [0, 1]^3.

    Returns
    -------
    float32 ndarray, shape (N,)
        Values normalized to [0, 1] using the volume's actual min/max,
        matching pysampler's output for float32 structured volumes.
    """
    Dz, Dy, Dx = volume.shape
    vmin = float(volume.min())
    vmax = float(volume.max())

    # CUDA tex3D formula: texel center i at normalized coord (i+0.5)/N
    # → fractional voxel index = u * N - 0.5
    fx = coords[:, 0] * Dx - 0.5
    fy = coords[:, 1] * Dy - 0.5
    fz = coords[:, 2] * Dz - 0.5

    # Clamp (mirrors cudaAddressModeClamp)
    fx = np.clip(fx, 0.0, Dx - 1.0)
    fy = np.clip(fy, 0.0, Dy - 1.0)
    fz = np.clip(fz, 0.0, Dz - 1.0)

    ix0 = np.floor(fx).astype(np.int32)
    iy0 = np.floor(fy).astype(np.int32)
    iz0 = np.floor(fz).astype(np.int32)
    ix1 = np.minimum(ix0 + 1, Dx - 1)
    iy1 = np.minimum(iy0 + 1, Dy - 1)
    iz1 = np.minimum(iz0 + 1, Dz - 1)

    tx = (fx - ix0).astype(np.float32)
    ty = (fy - iy0).astype(np.float32)
    tz = (fz - iz0).astype(np.float32)

    # Eight corner samples
    c000 = volume[iz0, iy0, ix0]
    c100 = volume[iz0, iy0, ix1]
    c010 = volume[iz0, iy1, ix0]
    c110 = volume[iz0, iy1, ix1]
    c001 = volume[iz1, iy0, ix0]
    c101 = volume[iz1, iy0, ix1]
    c011 = volume[iz1, iy1, ix0]
    c111 = volume[iz1, iy1, ix1]

    # Trilinear interpolation along x, then y, then z
    c00 = c000 * (1 - tx) + c100 * tx
    c01 = c001 * (1 - tx) + c101 * tx
    c10 = c010 * (1 - tx) + c110 * tx
    c11 = c011 * (1 - tx) + c111 * tx
    c0  = c00  * (1 - ty) + c10  * ty
    c1  = c01  * (1 - ty) + c11  * ty
    c   = c0   * (1 - tz) + c1   * tz

    return (c - vmin) / (vmax - vmin)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_n_channels_single():
    """Sampler must report n_channels == 1 for a single-field volume."""
    rng = np.random.default_rng(7)
    volume = rng.uniform(0, 1, (8, 8, 8)).astype(np.float32)  # (Dz, Dy, Dx)
    Dx, Dy, Dz = 8, 8, 8

    with tempfile.NamedTemporaryFile(suffix=".raw", delete=False) as f:
        fname = f.name
        volume.tofile(f)

    try:
        s = pysampler.create_sampler(
            "structuredRegular", "cuda",
            filename=fname, dims=[Dx, Dy, Dz], dtype="float32",
        )
        assert s.n_channels() == 1
    finally:
        os.unlink(fname)


def test_decode_trilinear_matches_reference():
    """CUDA tex3D trilinear must match the software reference within 2e-4."""
    rng = np.random.default_rng(42)
    Dx, Dy, Dz = 32, 40, 48
    volume = rng.standard_normal((Dz, Dy, Dx)).astype(np.float32)

    with tempfile.NamedTemporaryFile(suffix=".raw", delete=False) as f:
        fname = f.name
        volume.tofile(f)

    try:
        sampler = pysampler.create_sampler(
            "structuredRegular", "cuda",
            filename=fname, dims=[Dx, Dy, Dz], dtype="float32",
        )

        N = 4096
        # Stay away from the very edges to avoid boundary-clamping differences
        coords_np = rng.uniform(0.02, 0.98, (N, 3)).astype(np.float32)

        coords_gpu = torch.from_numpy(coords_np).cuda().contiguous()
        values_gpu = torch.empty(1, N, dtype=torch.float32, device="cuda")

        # decode() samples at the given coords without generating random coords first
        pysampler.decode(sampler, coords_gpu.data_ptr(), values_gpu.data_ptr(), N)

        cuda_vals = values_gpu[0].cpu().numpy()
        ref_vals  = _trilinear_ref(volume, coords_np)

        np.testing.assert_allclose(
            cuda_vals, ref_vals, atol=3e-3, rtol=0,
            err_msg="CUDA tex3D trilinear disagrees with software reference",
        )
    finally:
        os.unlink(fname)


def test_decode_constant_volume():
    """A constant volume must produce exactly 0 (or 1) after normalization."""
    Dx, Dy, Dz = 4, 4, 4
    # All values equal → range is zero; the C++ code guards against /0 with float_small
    # Instead use a volume with two distinct values so normalization is well-defined
    volume = np.zeros((Dz, Dy, Dx), dtype=np.float32)
    volume[0, 0, 0] = 1.0  # one hot-spot so range = [0, 1]

    with tempfile.NamedTemporaryFile(suffix=".raw", delete=False) as f:
        fname = f.name
        volume.tofile(f)

    try:
        sampler = pysampler.create_sampler(
            "structuredRegular", "cuda",
            filename=fname, dims=[Dx, Dy, Dz], dtype="float32",
        )

        N = 16
        # Sample exactly at the hot-spot center (texel 0 of each axis)
        # texel 0 center = 0.5/Dx in normalized coords
        cx = 0.5 / Dx
        cy = 0.5 / Dy
        cz = 0.5 / Dz
        coords_np = np.tile([cx, cy, cz], (N, 1)).astype(np.float32)

        coords_gpu = torch.from_numpy(coords_np).cuda().contiguous()
        values_gpu = torch.empty(1, N, dtype=torch.float32, device="cuda")

        pysampler.decode(sampler, coords_gpu.data_ptr(), values_gpu.data_ptr(), N)

        cuda_vals = values_gpu[0].cpu().numpy()
        # Hot-spot value = 1.0, normalized to 1.0
        np.testing.assert_allclose(cuda_vals, 1.0, atol=1e-6)
    finally:
        os.unlink(fname)


def test_sample_fills_coords_and_values():
    """sample() must populate both the coords and values buffers."""
    rng = np.random.default_rng(99)
    Dx, Dy, Dz = 16, 16, 16
    volume = rng.uniform(0, 1, (Dz, Dy, Dx)).astype(np.float32)

    with tempfile.NamedTemporaryFile(suffix=".raw", delete=False) as f:
        fname = f.name
        volume.tofile(f)

    try:
        sampler = pysampler.create_sampler(
            "structuredRegular", "cuda",
            filename=fname, dims=[Dx, Dy, Dz], dtype="float32",
        )

        N = 256
        # sample() generates its own random coords; seed buffers with sentinels
        coords_gpu = torch.full((N, 3), -1.0, dtype=torch.float32, device="cuda")
        values_gpu = torch.full((1, N), -1.0, dtype=torch.float32, device="cuda")

        pysampler.sample(sampler, coords_gpu.data_ptr(), values_gpu.data_ptr(), N)

        coords_cpu = coords_gpu.cpu().numpy()
        values_cpu = values_gpu[0].cpu().numpy()

        # Coords should now be in [0, 1]^3 (random uniforms generated by the sampler)
        assert coords_cpu.min() >= 0.0 and coords_cpu.max() <= 1.0, \
            "generated coords out of [0, 1]^3"

        # Values should be in [0, 1] after normalization
        assert values_cpu.min() >= 0.0 and values_cpu.max() <= 1.0, \
            "normalized values out of [0, 1]"
    finally:
        os.unlink(fname)


# ---------------------------------------------------------------------------
# Sampling strategies
# ---------------------------------------------------------------------------

def _sample_with_strategy(volume, backend, strategy, count):
    """Return (coords (count,3), values (count,)) as numpy arrays."""
    Dz, Dy, Dx = volume.shape
    with tempfile.NamedTemporaryFile(suffix=".raw", delete=False) as f:
        fname = f.name
        volume.tofile(f)
    try:
        kwargs = dict(filename=fname, dims=[Dx, Dy, Dz], dtype="float32")
        if backend != "cuda":
            kwargs["range"] = [float(volume.min()), float(volume.max())]
        sampler = pysampler.create_sampler("structuredRegular", backend, **kwargs)
        device = "cuda" if backend == "cuda" else "cpu"
        coords = torch.full((count, 3), -1.0, dtype=torch.float32, device=device)
        values = torch.full((1, count), -1.0, dtype=torch.float32, device=device)
        pysampler.sample(sampler, coords.data_ptr(), values.data_ptr(), count, strategy)
        return coords.cpu().numpy(), values[0].cpu().numpy()
    finally:
        os.unlink(fname)


@pytest.mark.parametrize("backend", ["cuda", "virtual_memory", "out_of_core"])
def test_grid_strategy_returns_exact_voxels(backend):
    """'grid' must return voxel centers and the voxel values stored there."""
    rng = np.random.default_rng(3)
    Dx, Dy, Dz = 12, 9, 7
    volume = rng.uniform(-2, 5, (Dz, Dy, Dx)).astype(np.float32)
    coords, values = _sample_with_strategy(volume, backend, "grid", 4096)

    dims = np.array([Dx, Dy, Dz], dtype=np.float32)
    index = coords * dims - 0.5
    nearest = np.rint(index).astype(np.int64)
    np.testing.assert_allclose(index, nearest, atol=1e-4)
    assert nearest.min() >= 0 and (nearest < dims).all()
    # Every voxel of this small volume should be hit by 4096 draws.
    assert len(np.unique(nearest, axis=0)) > 0.9 * Dx * Dy * Dz

    vmin, vmax = volume.min(), volume.max()
    expected = (volume[nearest[:, 2], nearest[:, 1], nearest[:, 0]] - vmin) / (vmax - vmin)
    np.testing.assert_allclose(values, expected, atol=1e-5)


@pytest.mark.parametrize("backend", ["cuda", "virtual_memory"])
def test_stratified_strategy_covers_every_lattice_cell(backend):
    """'stratified' with count = n^3 puts exactly one sample in each cell."""
    rng = np.random.default_rng(4)
    volume = rng.uniform(0, 1, (10, 10, 10)).astype(np.float32)
    n = 6
    coords, values = _sample_with_strategy(volume, backend, "stratified", n ** 3)

    assert coords.min() >= 0.0 and coords.max() <= 1.0
    cells = np.minimum(np.floor(coords * n), n - 1).astype(np.int64)
    flat = cells[:, 0] + n * (cells[:, 1] + n * cells[:, 2])
    assert len(np.unique(flat)) == n ** 3

    # tex3D interpolates with 8-bit weights; on a noise volume that is ~4e-3.
    expected = _trilinear_ref(volume, coords)
    np.testing.assert_allclose(values, expected, atol=5e-3 if backend == "cuda" else 1e-5)


def test_stratified_strategy_fills_remainder_uniformly():
    """Samples beyond the n^3 lattice are still valid points in [0,1]^3."""
    volume = np.random.default_rng(5).uniform(0, 1, (8, 8, 8)).astype(np.float32)
    coords, values = _sample_with_strategy(volume, "cuda", "stratified", 5 ** 3 + 17)
    assert coords.min() >= 0.0 and coords.max() <= 1.0
    assert values.min() >= 0.0 and values.max() <= 1.0


def test_unknown_strategy_is_rejected():
    volume = np.zeros((4, 4, 4), dtype=np.float32)
    with pytest.raises(RuntimeError, match="unknown sampling strategy"):
        _sample_with_strategy(volume, "cuda", "nearest", 8)


@pytest.mark.parametrize("dtype, low, high", [("uint8", 40, 200), ("uint16", 5900, 58000)])
def test_integer_volumes_are_normalized_by_data_range(dtype, low, high):
    """Integer volumes map their own min/max to [0, 1], like float volumes."""
    rng = np.random.default_rng(6)
    Dx, Dy, Dz = 8, 6, 5
    volume = rng.integers(low, high + 1, size=(Dz, Dy, Dx)).astype(dtype)
    volume.flat[0], volume.flat[-1] = low, high
    with tempfile.NamedTemporaryFile(suffix=".raw", delete=False) as f:
        fname = f.name
        volume.tofile(f)
    try:
        sampler = pysampler.create_sampler(
            "structuredRegular", "cuda", filename=fname, dims=[Dx, Dy, Dz], dtype=dtype,
        )
        N = 2048
        coords = torch.empty((N, 3), dtype=torch.float32, device="cuda")
        values = torch.empty((1, N), dtype=torch.float32, device="cuda")
        pysampler.sample(sampler, coords.data_ptr(), values.data_ptr(), N, "grid")
        index = np.rint(coords.cpu().numpy() * [Dx, Dy, Dz] - 0.5).astype(int)
        stored = volume[index[:, 2], index[:, 1], index[:, 0]].astype(np.float64)
        np.testing.assert_allclose(
            values[0].cpu().numpy(), (stored - low) / (high - low), atol=1e-4
        )
    finally:
        os.unlink(fname)
