#include "cuda_launch_config.hpp"
#include <thrust/device_vector.h>
#include <cassert>

__global__ void saxpy(float a, float *x, float *y, size_t n)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if(i < n)
  {
    y[i] = a * x[i] + y[i];
  }
}

int main()
{
  size_t n = 1 << 20;
  thrust::device_vector<float> x(n, 10);
  thrust::device_vector<float> y(n, 100);

  float a = 10;

  // we'd like to launch saxpy, but we're not sure which block size to use
  // let's use a heuristic which promotes occupancy.

  // first, get the cudaFuncAtttributes object corresponding to saxpy
  cudaFuncAttributes attributes;
  cudaFuncGetAttributes(&attributes, saxpy);

  // next, get the cudaDeviceProp object corresponding to the current device
  int device;
  cudaGetDevice(&device);

  cudaDeviceProp properties;
  cudaGetDeviceProperties(&properties, device);

  // we can combine the two to compute a block size
  size_t num_threads = block_size_with_maximum_potential_occupancy(attributes, properties);

  // compute the number of blocks of size num_threads to launch
  size_t num_blocks = n / num_threads;

  // check for partial block at the end
  if(n % num_threads) ++num_blocks; 

  saxpy<<<num_blocks,num_threads>>>(a, raw_pointer_cast(x.data()), raw_pointer_cast(y.data()), n);

  // validate the result
  thrust::device_vector<float> reference(n, 200);

  assert(reference == y);

  return 0;
}

