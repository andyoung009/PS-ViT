// 这段代码是一个CUDA的helper文件，包含了一些CUDA的宏定义和函数。其中，CUDA_1D_KERNEL_LOOP是一个宏定义，用于在CUDA kernel中进行循环迭代。
#ifndef CUDA_HELPER
#define CUDA_HELPER

#include <cuda.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <THC/THCAtomics.cuh>

using at::Half;
using at::Tensor;
using phalf = at::Half;

#define __PHALF(x) (x)
// 这些函数都是在CUDA中执行的，因此需要使用CUDA的数据类型和函数。此外，该文件还包含了一些头文件和命名空间的引用，如cuda.h、ATen、THC等。
// 这些头文件和命名空间提供了在CUDA中使用的一些函数和数据类型。

// 这是一个 CUDA 宏，用于定义一个循环，遍历一个一维线程网格。 
// 该宏接受两个参数：`i` 是循环变量，`n` 是循环迭代次数。 
// 循环使用线程索引 `blockIdx.x`、块维度 `blockDim.x` 和块内线程索引 `threadIdx.x` 遍历一维线程网格。
// 循环从 `blockIdx.x * blockDim.x + threadIdx.x` 开始，每次增加 `blockDim.x * gridDim.x`，直到达到 `(n)`。
// 这个宏通常用于 CUDA 内核中，以跨多个线程并行计算。通过使用这样的循环，每个线程可以执行计算的不同部分，从而实现更快、更高效的执行。
#define CUDA_1D_KERNEL_LOOP(i, n)                                \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
        i += blockDim.x * gridDim.x)

#define THREADS_PER_BLOCK 512

// GET_BLOCKS是一个函数，用于计算线程块的数量。
inline int GET_BLOCKS(const int N)
{
    int optimal_block_num = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    int max_block_num = 4096;
    return min(optimal_block_num, max_block_num);
}

// bilinear_interpolate和bilinear_interpolate_gradient是两个函数，用于进行双线性插值和双线性插值梯度计算。
template <typename T>
__device__ T bilinear_interpolate(const T *input,
                                  const int height,
                                  const int width,
                                  T h,
                                  T w)
{
    if (h <= -1 || h >= height || w <= -1 || w >= width)
    {
        return 0;
    }

    int h_low = floor(h);
    int w_low = floor(w);
    int h_high = h_low + 1;
    int w_high = w_low + 1;

    T lh = h - h_low;
    T lw = w - w_low;
    T hh = 1. - lh;
    T hw = 1. - lw;

    T v1 = 0;
    if (h_low >= 0 && w_low >= 0)
        v1 = input[h_low * width + w_low];
    T v2 = 0;
    if (h_low >= 0 && w_high <= width - 1)
        v2 = input[h_low * width + w_high];
    T v3 = 0;
    if (h_high <= height - 1 && w_low >= 0)
        v3 = input[h_high * width + w_low];
    T v4 = 0;
    if (h_high <= height - 1 && w_high <= width - 1)
        v4 = input[h_high * width + w_high];

    T w1 = hh * hw;
    T w2 = hh * lw;
    T w3 = lh * hw;
    T w4 = lh * lw;

    T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
    return val;
}

// bilinear_interpolate和bilinear_interpolate_gradient是两个函数，用于进行双线性插值和双线性插值梯度计算。
template <typename T>
__device__ void bilinear_interpolate_gradient(const int height,
                                              const int width,
                                              T y,
                                              T x,
                                              T& w1,
                                              T& w2,
                                              T& w3,
                                              T& w4,
                                              int& y_low,
                                              int& y_high,
                                              int& x_low,
                                              int& x_high)
{
    if (y <= -1. || y >= height || x <= -1. || x >= width)
    {
        w1 = w2 = w3 = w4 = 0.;
        x_low = x_high = y_low = y_high = -1;
        return;
    }

    if (y <= 0) y = 0;
    if (x <= 0) x = 0;

    y_low = (int) y;
    x_low = (int) x;

    if (y_low >= height - 1)
    {
        y_high = y_low = height - 1;
        y = (T) y_low;
    }
    else
    {
        y_high = y_low + 1;
    }

    if (x_low >= width - 1)
    {
        x_high = x_low = width - 1;
        x = (T) x_low;
    }
    else
    {
        x_high = x_low + 1;
    }

    T ly = y - y_low;
    T lx = x - x_low;
    T hy = 1. - ly;
    T hx = 1. - lx;

    w1 = hy * hx;
    w2 = hy * lx;
    w3 = ly * hx;
    w4 = ly * lx;
}

#endif  // CUDA_HELPER
