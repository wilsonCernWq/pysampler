// Coordinate rewrites behind the "grid" and "stratified" sampling strategies.
// Both take `count` uniform random points in [0,1]^3 and rewrite them in place;
// SamplerBase::sample_grid / sample_stratified then decode at those points.

#include "config.h"
#include "sampler.h"

#include <cuda/cuda_utils.h>

#include <cmath>

namespace pysampler {

namespace {

// Random voxel index along one axis -> its center in normalized coordinates.
// curand returns values in (0,1], hence the clamp.
__host__ __device__ inline float voxel_center(float u, int n) {
  const float i = fminf(floorf(u * (float)n), (float)(n - 1));
  return (i + 0.5f) / (float)n;
}

// Sample `m` of an n^3 lattice: uniform within its own cell. Samples past the
// lattice (m >= n^3) stay uniform over the whole domain.
__host__ __device__ inline vec3f lattice_jitter(vec3f u, size_t m, size_t n) {
  if (m >= n * n * n) return u;
  const float rn = 1.f / (float)n;
  const vec3f cell((float)(m % n), (float)((m / n) % n), (float)(m / (n * n)));
  // Keep the point strictly inside its cell (u can be exactly 1).
  const float inside = 0.99999994f;
  return vec3f(fminf((cell.x + u.x) * rn, (cell.x + 1.f) * rn * inside),
               fminf((cell.y + u.y) * rn, (cell.y + 1.f) * rn * inside),
               fminf((cell.z + u.z) * rn, (cell.z + 1.f) * rn * inside));
}

inline size_t lattice_resolution(size_t count) {
  size_t n = (size_t)std::cbrt((double)count);
  while ((n + 1) * (n + 1) * (n + 1) <= count) ++n;
  while (n > 0 && n * n * n > count) --n;
  return n;
}

}

void snap_to_voxel_centers(vec3f* coords, size_t count, const vec3i& dims, Device device) {
  const int nx = dims.x, ny = dims.y, nz = dims.z;
  if (device == Device::CUDA) {
    util::parallel_for_gpu(count, [coords, nx, ny, nz] __device__ (size_t i) {
      const vec3f u = coords[i];
      coords[i] = vec3f(voxel_center(u.x, nx), voxel_center(u.y, ny), voxel_center(u.z, nz));
    });
  }
  else {
    for (size_t i = 0; i < count; ++i) {
      const vec3f u = coords[i];
      coords[i] = vec3f(voxel_center(u.x, nx), voxel_center(u.y, ny), voxel_center(u.z, nz));
    }
  }
}

void jitter_within_lattice(vec3f* coords, size_t count, Device device) {
  const size_t n = lattice_resolution(count);
  if (n == 0) return;
  if (device == Device::CUDA) {
    util::parallel_for_gpu(count, [coords, n] __device__ (size_t i) {
      coords[i] = lattice_jitter(coords[i], i, n);
    });
  }
  else {
    for (size_t i = 0; i < count; ++i) coords[i] = lattice_jitter(coords[i], i, n);
  }
}

}
