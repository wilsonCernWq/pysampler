#pragma once

#include "generate_random_numbers.h"

#include <gdt/math/vec.h>
#include <gdt/math/box.h>
#include <gdt/math/mat.h>

#include <vector>
#include <memory>

namespace pysampler {

using gdt::vec3i;
using gdt::vec3f;
using gdt::box3f;
using gdt::range1f;
using gdt::affine3f;

enum dtype {
  UINT8, UINT16, UINT32,
  INT8,  INT16,  INT32,
  FLOAT, DOUBLE
};

template <typename T>
static inline T div_round_up(T val, T divisor) {
	return (val + divisor - 1) / divisor;
}

template <typename T>
static inline T next_multiple(T val, T divisor) {
	return div_round_up(val, divisor) * divisor;
}

static inline int sizeof_type(dtype type) {
  switch (type) {
  case UINT8:
  case INT8:   return sizeof(char);
  case UINT16:
  case INT16:  return sizeof(short);
  case UINT32:
  case INT32:  return sizeof(int);
  case FLOAT:  return sizeof(float);
  case DOUBLE: return sizeof(double);
  default: throw std::runtime_error("unknown data type");
  }
}

static inline float read_buffer(char* buffer, size_t id, dtype type) {
  switch (type) {
  case UINT8:  return float(((uint8_t* )buffer)[id]);
  case INT8:   return float(((int8_t*  )buffer)[id]);
  case UINT16: return float(((uint16_t*)buffer)[id]);
  case INT16:  return float(((int16_t* )buffer)[id]); 
  case UINT32: return float(((uint32_t*)buffer)[id]);
  case INT32:  return float(((int32_t* )buffer)[id]); 
  case FLOAT:  return float(((float*   )buffer)[id]);   
  case DOUBLE: return float(((double*  )buffer)[id]);  
  default: throw std::runtime_error("unknown data type"); 
  }
}

struct SamplerBase {
public:
  virtual ~SamplerBase() {}
  virtual pysampler::Device device() const = 0;
  virtual pysampler::box3f bounds() const = 0;
  virtual int n_channels() const = 0;
  virtual void sample(void* d_coords, void* d_values, const vec3f& lower, const vec3f& upper, size_t count) = 0;
  virtual void decode1d(void* d_coords, void* d_values, const vec3f& lower, const vec3f& upper, size_t count) = 0;
  virtual void decode3d(void* d_coords, void* d_values, const vec3f& lower, const vec3f& upper, const vec3i& dims) = 0;

  // Voxel resolution of the underlying structured grid. Volume types without
  // one keep the default, which makes grid sampling unavailable for them.
  virtual vec3i dims() const { throw std::runtime_error("grid sampling requires a structured regular volume"); }

  // Alternative sampling strategies. Same buffers and layout as sample().
  //   sample_grid       : random voxels (with replacement); coords are voxel
  //                       centers (i + 0.5) / dims, values are the stored voxels.
  //   sample_stratified : jittered lattice. [0,1]^3 is split into n^3 cells,
  //                       n = floor(cbrt(count)), with one uniform sample per
  //                       cell; the remaining count - n^3 samples are uniform.
  void sample_grid(void* d_coords, void* d_values, const vec3f& lower, const vec3f& upper, size_t count);
  void sample_stratified(void* d_coords, void* d_values, const vec3f& lower, const vec3f& upper, size_t count);
};

// In-place rewrites of `count` uniform random points in [0,1]^3 (host or CUDA
// pointer, selected by `device`). Implemented in sample_strategies.cu.
void snap_to_voxel_centers(vec3f* coords, size_t count, const vec3i& dims, Device device);
void jitter_within_lattice(vec3f* coords, size_t count, Device device);

typedef std::shared_ptr<SamplerBase> sampler_t;

// ------------------------------------------------------------------
// TODO support all the volume types
//    [x] structured regular volume
//    [x] OpenVDB volume
//    [x] ExaBrick/ExaStitch volume
//    [ ] AMR volume
//    [ ] unstructured volume
// ------------------------------------------------------------------

struct _VolumeDescStructured {
  dtype type;
  vec3i dims;
  vec3f spacing = vec3f{ 1, 1, 1 };
  int n_channels = 1;
  std::vector<range1f> ranges;
};

struct VolumeFileStructured : _VolumeDescStructured {
  const char* filename;
  unsigned long long offset = 0;
  bool is_big_endian = false;
};

struct VolumeDataStructured : _VolumeDescStructured {
  const uint8_t * data;
};

struct VolumeFileOpenVDB {
  const char* filename;
  const char* field;
};

struct VolumeFileExaBrick {
  const char* bricks;
  const char* scalar;
  const char* kdtree = nullptr;
};

struct VolumeFileExaStitch {
  const char* umesh;
  const char* grids;
  const char* scalar;
};

struct VolumeFileVTKm {
  const char* field;
  std::vector<const char*> files;
};

sampler_t create_sampler(const char* device, VolumeFileStructured desc);
sampler_t create_sampler(const char* device, VolumeDataStructured desc);
sampler_t create_sampler(const char* device, VolumeFileOpenVDB desc);
sampler_t create_sampler(const char* device, VolumeFileExaBrick desc);
sampler_t create_sampler(const char* device, VolumeFileExaStitch desc);
sampler_t create_sampler(const char* device, VolumeFileVTKm desc);

std::shared_ptr<char[]> load_regular_grid(const char* filename, vec3i dims, int n_channels, dtype type, size_t offset, bool is_big_endian);

} // namespace pysampler
