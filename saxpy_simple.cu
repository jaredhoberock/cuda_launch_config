#include "cuda_launch_config.hpp"
#include <thrust/device_vector.h>
#include <thrust/logical.h>
#include <thrust/functional.h>
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

  size_t num_threads = block_size_with_maximum_potential_occupancy(saxpy);

  // compute the number of blocks of size num_threads to launch
  size_t num_blocks = n / num_threads;

  // check for partial block at the end
  if(n % num_threads) ++num_blocks; 

  saxpy<<<num_blocks,num_threads>>>(a, raw_pointer_cast(x.data()), raw_pointer_cast(y.data()), n);

  // validate the result
  assert(thrust::all_of(y.begin(), y.end(), thrust::placeholders::_1 == 200));

  return 0;
}

