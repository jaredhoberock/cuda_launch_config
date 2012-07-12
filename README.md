It is often the case in CUDA programming that we wish to launch a kernel but don't know which block size to use.
Perhaps the kernel is an essential piece of our algorithm, but its performance is not important enough to warrant a thorough tuning.
Perhaps we aren't able to benchmark the kernel because it is a library function whose runtime behavior isn't known at authorship time.
Perhaps we're simply in a hurry and have more interesting things to work on.

```
__global__ void my_kernel();

int main()
{
  // what goes here?
  my_kernel<<<???>>>();
  return 0;
}
```

Whatever the case, we need to choose some block size because it is necessary to launch the kernel.

We'd like a block size which is both

  1. guaranteed to fit within the resource constraints of the device and
  2. does not result in a kernel launch with pathological performance

One way to choose a block size is to use a heuristic which promotes
utilization, or "occupancy".  A kernel with high potential occupancy can often
perform well, and isn't likely to perform pathologically slow. So, without
further information beyond a kernel's resource requirements and the GPU of
interest, high occupancy is a reasonable goal.

The functions included in this library compute CUDA block sizes which are intended to promote high occupancy for a CUDA kernel.

Here's how to use it for `saxpy`:

```
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
  assert(thrust::all_of(y.begin(), y.end(), thrust::placeholders::_1 == 200));

  return 0;
}

```

