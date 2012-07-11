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
#include <thrust/tuple.h>
#include <thrust/extrema.h>

namespace __cuda_launch_config_detail
{

using std::size_t;

namespace util
{


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
inline size_t smem_allocation_unit(const cudaDeviceProp &)
{
  return 512;
}


// granularity of register allocation
inline size_t reg_allocation_unit(const cudaDeviceProp &properties)
{
  return (properties.major < 2 && properties.minor < 2) ? 256 : 512;
}


// granularity of warp allocation
inline size_t warp_allocation_multiple(const cudaDeviceProp &)
{
  return 2;
}


inline size_t max_blocks_per_multiprocessor(const cudaDeviceProp &)
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

  return thrust::min<size_t>(ctaLimitRegs, thrust::min<size_t>(ctaLimitSMem, thrust::min<size_t>(ctaLimitThreads, maxBlocksPerSM)));
}


inline __host__ __device__
size_t proportional_smem_allocation(const cudaDeviceProp     &properties,
                                    const cudaFuncAttributes &attributes,
                                    size_t blocks_per_processor)
{
  size_t smem_per_processor    = properties.sharedMemPerBlock;
  size_t smem_allocation_unit  = smem_allocation_unit(properties);

  size_t total_smem_per_block  = util::round_z(smem_per_processor / blocks_per_processor, smem_allocation_unit);
  size_t static_smem_per_block = attributes.sharedSizeBytes;
  
  return total_smem_per_block - static_smem_per_block;
}


template <typename UnaryFunction>
inline __host__ __device__
thrust::pair<size_t,size_t> default_block_configuration(const cudaDeviceProp     &properties,
                                                        const cudaFuncAttributes &attributes,
                                                        UnaryFunction block_size_to_smem_size)
{
  size_t max_occupancy      = properties.maxThreadsPerMultiProcessor;
  size_t largest_blocksize  = (thrust::min)(properties.maxThreadsPerBlock, attributes.maxThreadsPerBlock);
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

  return thrust::make_pair(max_blocksize, max_occupancy / max_blocksize);
}


} // end namespace __cuda_launch_config_detail


template<typename UnaryFunction>
inline __host__ __device__
thrust::tuple<std::size_t, std::size_t, std::size_t>
block_size_with_maximum_potential_occupancy(const cudaFuncAttributes &attributes,
                                            const cudaDeviceProp &properties,
                                            UnaryFunction block_size_to_dynamic_smem_size)
{
  using thrust::tie;
  using __cuda_launch_config_detail;

  std::size_t block_size = 0, blocks_per_multiprocessor = 0;

  tie(block_size, blocks_per_multiprocessor) = default_block_configuration(properties, attributes, block_size_to_dynamic_smem_size);

  return thrust::make_tuple(block_size * blocks_per_multiprocessor, block_size_to_dynamic_smem_size(block_size));
}


inline __host__ __device__
thrust::tuple<std::size_t, std::size_t, std::size_t>
block_size_with_maximum_potential_occupancy(const cudaFuncAttributes &attributes,
                                            const cudaDeviceProp &properties)
{
  using __cuda_launch_config_detail;
  return block_size_with_maximum_potential_occupancy(attributes, properties, util::zero_function<std::size_t>());
}


inline __host__ __device__
thrust::tuple<std::size_t, std::size_t, std::size_t>
block_size_with_maximum_potential_occupancy_subject_to_available_smem(const cudaFuncAttributes &attributes,
                                                                      const cudaDeviceProp &properties)
{
  using thrust::tie;
  using __cuda_launch_config_detail;

  std::size_t block_size = 0, blocks_per_multiprocessor = 0;

  tie(block_size, blocks_per_multiprocessor) = default_block_configuration(properties, attributes);

  size_t smem_per_block = proportional_smem_allocation(properties, attributes, config.second);

  return thrust::make_tuple(block_size, blocks_per_multiprocessor, smem_per_block);
}

