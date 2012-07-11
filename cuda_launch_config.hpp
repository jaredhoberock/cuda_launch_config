/*
 *  Copyright 2008-2012 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#pragma once

#include <cstddef>
#include <cuda_runtime_api.h>


/*! Computes a block size in number of threads for a CUDA kernel using a occupancy-promoting heuristic.
 *  \param attributes The cudaFuncAttributes corresponding to a __global__ function of interest on a GPU of interest.
 *  \param properties The cudaDeviceProp corresponding to a GPU on which to launch the __global__ function of interest.
 *  \return A CUDA block size, in number of threads, which the resources of the GPU's streaming multiprocessor can
 *          accomodate and which is intended to promote occupancy. The result is equivalent to the one performed by
 *          the "CUDA Occupancy Calculator". 
 *  \note The __global__ function of interest is presumed to use 0 bytes of dynamically-allocated __shared__ memory.
 */
inline __host__ __device__
std::size_t block_size_with_maximum_potential_occupancy(const cudaFuncAttributes &attributes,
                                                        const cudaDeviceProp &properties);

/*! Computes a block size in number of threads for a CUDA kernel using a occupancy-promoting heuristic.
 *  Use this version of the function when a CUDA block's dynamically-allocated __shared__ memory requirements
 *  vary with the size of the block.
 *  \param attributes The cudaFuncAttributes corresponding to a __global__ function of interest on a GPU of interest.
 *  \param properties The cudaDeviceProp corresponding to a GPU on which to launch the __global__ function of interest.
 *  \param block_size_to_dynamic_smem_bytes A unary function which maps an integer CUDA block size to the number of bytes
 *         of dynamically-allocated __shared__ memory required by a CUDA block of that size.
 *  \return A CUDA block size, in number of threads, which the resources of the GPU's streaming multiprocessor can
 *          accomodate and which is intended to promote occupancy. The result is equivalent to the one performed by
 *          the "CUDA Occupancy Calculator". 
 */
template<typename UnaryFunction>
inline __host__ __device__
std::size_t block_size_with_maximum_potential_occupancy(const cudaFuncAttributes &attributes,
                                                        const cudaDeviceProp &properties,
                                                        UnaryFunction block_size_to_dynamic_smem_size);



namespace __cuda_launch_config_detail
{

using std::size_t;

namespace util
{


template<typename T>
inline __host__ __device__
T min_(const T &lhs, const T &rhs)
{
  return rhs < lhs ? rhs : lhs;
}


template <typename T>
struct zero_function
{
  inline __host__ __device__
  T operator()(T)
  {
    return 0;
  }
};


// x/y rounding towards +infinity for integers, used to determine # of blocks/warps etc.
template<typename L, typename R>
  inline __host__ __device__ L divide_ri(const L x, const R y)
{
    return (x + (y - 1)) / y;
}

// x/y rounding towards zero for integers, used to determine # of blocks/warps etc.
template<typename L, typename R>
  inline __host__ __device__ L divide_rz(const L x, const R y)
{
    return x / y;
}

// round x towards infinity to the next multiple of y
template<typename L, typename R>
  inline __host__ __device__ L round_i(const L x, const R y){ return y * divide_ri(x, y); }

// round x towards zero to the next multiple of y
template<typename L, typename R>
  inline __host__ __device__ L round_z(const L x, const R y){ return y * divide_rz(x, y); }

} // end namespace util



// granularity of shared memory allocation
inline __host__ __device__
size_t smem_allocation_unit(const cudaDeviceProp &)
{
  return 512;
}


// granularity of register allocation
inline __host__ __device__
size_t reg_allocation_unit(const cudaDeviceProp &properties)
{
  return (properties.major < 2 && properties.minor < 2) ? 256 : 512;
}


// granularity of warp allocation
inline __host__ __device__
size_t warp_allocation_multiple(const cudaDeviceProp &)
{
  return 2;
}


inline __host__ __device__
size_t max_blocks_per_multiprocessor(const cudaDeviceProp &)
{
  return 8;
}


inline __host__ __device__
size_t max_active_blocks_per_multiprocessor(const cudaDeviceProp     &properties,
                                            const cudaFuncAttributes &attributes,
                                            size_t CTA_SIZE,
                                            size_t dynamic_smem_bytes)
{
  // Determine the maximum number of CTAs that can be run simultaneously per SM
  // This is equivalent to the calculation done in the CUDA Occupancy Calculator spreadsheet
  const size_t regAllocationUnit      = reg_allocation_unit(properties);
  const size_t warpAllocationMultiple = warp_allocation_multiple(properties);
  const size_t smemAllocationUnit     = smem_allocation_unit(properties);
  const size_t maxThreadsPerSM        = properties.maxThreadsPerMultiProcessor;  // 768, 1024, 1536, etc.
  const size_t maxBlocksPerSM         = max_blocks_per_multiprocessor(properties);

  // Number of warps (round up to nearest whole multiple of warp size & warp allocation multiple)
  const size_t numWarps = util::round_i(util::divide_ri(CTA_SIZE, properties.warpSize), warpAllocationMultiple);

  // Number of regs is regs per thread times number of warps times warp size
  const size_t regsPerCTA = util::round_i(attributes.numRegs * properties.warpSize * numWarps, regAllocationUnit);

  const size_t smemBytes  = attributes.sharedSizeBytes + dynamic_smem_bytes;
  const size_t smemPerCTA = util::round_i(smemBytes, smemAllocationUnit);

  const size_t ctaLimitRegs    = regsPerCTA > 0 ? properties.regsPerBlock      / regsPerCTA : maxBlocksPerSM;
  const size_t ctaLimitSMem    = smemPerCTA > 0 ? properties.sharedMemPerBlock / smemPerCTA : maxBlocksPerSM;
  const size_t ctaLimitThreads =                  maxThreadsPerSM              / CTA_SIZE;

  return util::min_(ctaLimitRegs, util::min_(ctaLimitSMem, util::min_(ctaLimitThreads, maxBlocksPerSM)));
}


template <typename UnaryFunction>
inline __host__ __device__
size_t default_block_size(const cudaDeviceProp     &properties,
                          const cudaFuncAttributes &attributes,
                          UnaryFunction block_size_to_smem_size)
{
  size_t max_occupancy      = properties.maxThreadsPerMultiProcessor;
  size_t largest_blocksize  = util::min_(properties.maxThreadsPerBlock, attributes.maxThreadsPerBlock);
  size_t granularity        = properties.warpSize;
  size_t max_blocksize      = 0;
  size_t highest_occupancy  = 0;

  for(size_t blocksize = largest_blocksize; blocksize != 0; blocksize -= granularity)
  {
    size_t occupancy = blocksize * max_active_blocks_per_multiprocessor(properties, attributes, blocksize, block_size_to_smem_size(blocksize));

    if(occupancy > highest_occupancy)
    {
      max_blocksize = blocksize;
      highest_occupancy = occupancy;
    }

    // early out, can't do better
    if(highest_occupancy == max_occupancy)
      break;
  }

  return max_blocksize;
}


} // end namespace __cuda_launch_config_detail


template<typename UnaryFunction>
inline __host__ __device__
std::size_t block_size_with_maximum_potential_occupancy(const cudaFuncAttributes &attributes,
                                                        const cudaDeviceProp &properties,
                                                        UnaryFunction block_size_to_dynamic_smem_size)
{
  return __cuda_launch_config_detail::default_block_size(properties, attributes, block_size_to_dynamic_smem_size);
}


inline __host__ __device__
std::size_t block_size_with_maximum_potential_occupancy(const cudaFuncAttributes &attributes,
                                                        const cudaDeviceProp &properties)
{
  return block_size_with_maximum_potential_occupancy(attributes, properties, __cuda_launch_config_detail::util::zero_function<std::size_t>());
}

