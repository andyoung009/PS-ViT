#ifndef PROGRESSIVE_SAMPLING_CUDA_KERNEL
#define PROGRESSIVE_SAMPLING_CUDA_KERNEL

#include "cuda_helper.hpp"

// 这段代码是一个CUDA kernel函数，用于执行progressive sampling的前向传播操作。它的输入包括一个输入张量、一个点张量、一个偏移张量和一个gamma值，以及一些维度参数。
// 输出是一个采样后的张量。该函数使用CUDA1DKERNELLOOP宏来并行处理输入张量中的每个元素。
// 在每个线程中，它首先计算当前点和偏移量的加权和，然后使用双线性插值从输入张量中获取采样值，并将其存储在输出张量中。
template <typename T>
__global__ void progressive_sampling_forward_cuda_kernel(const int nthreads,
                                                         const T* input,
                                                         const T* point,
                                                         const T* offset,
                                                         T* output,
                                                         const int channels,
                                                         const int point_num,
                                                         const int height,
                                                         const int width,
                                                         const T gamma)
{   
    // 这段代码是一个 CUDA 的 1D kernel，其中包含了一个循环，使用了三个变量：index、nthreads、和 channels。这段代码的作用是对输入的数据进行双线性插值，输出一个新的张量。
    CUDA_1D_KERNEL_LOOP(index, nthreads)
    {   
        // 具体来说，这段代码对于输出张量中的每一个元素，都会计算其对应的输入张量中的位置，并进行双线性插值。其中，index 变量表示当前元素在输出张量中的索引，nthreads 表示输出张量的总大小，而 channels 则表示输入张量中的通道数。
        // 在循环中，代码首先通过取余操作计算出了当前元素在第几个通道上，第几个点上，以及属于哪一个样本。然后，通过这些信息，我们可以确定当前元素在输入张量中的位置。
        // 接着，代码通过从输入张量、偏移量张量和当前点的坐标计算出在输入张量中的位置，然后使用 bilinear_interpolate 函数对其进行双线性插值，最终得到当前输出张量元素的值。
        // 其中，gamma 是一个缩放因子，用于控制偏移量的大小。总的来说，这段代码的作用是实现了一个带有偏移量的双线性插值操作，用于对输入张量进行处理，得到输出张量。
        int c = index % channels;
        int p = (index / channels) % point_num;
        int n = index / channels / point_num;

        const T* current_point = point + (n * point_num + p) * 2;
        const T* current_offset = offset + (n * point_num + p) * 2;
        const T* current_input = input + (n * channels + c) * height * width;

        // 在这段代码中，`T`是一个模板参数，用于表示数据类型。它可以是`float`、`double`等类型。`y`是一个变量名，表示当前点在输入张量中的纵向位置。
        // 由于`T`是一个模板参数，所以`y`的数据类型也是`T`。因此，`const T y`表示定义了一个类型为`T`的常量`y`。
        const T y = current_point[0] + current_offset[0] * gamma;
        const T x = current_point[1] + current_offset[1] * gamma;

        // 根据提供的代码文件路径 /d:/Work/PS-ViT/layers/cinclude/progressive_sampling_cuda_kernel.cuh
        // 可以在该文件中找到 bilinearinterpolate 函数的定义。该函数定义在该文件顶部的 #include "cuda_helper.hpp" 头文件中。
        // 该函数可用于进行双线性插值，以从输入张量中获取采样值。因为该函数的定义被包含在CUDA kernel代码文件中，所以在 kernel 函数中可以直接调用。
        output[index] = bilinear_interpolate(current_input, height, width, y, x);
    }
}


template <typename T>
__global__ void progressive_sampling_backward_cuda_kernel(const int nthreads,
                                                          const T* grad_output,
                                                          const T* input,
                                                          const T* point,
                                                          const T* offset,
                                                          T* grad_input,
                                                          T* grad_offset,
                                                          int channels,
                                                          int point_num,
                                                          int height,
                                                          int width,
                                                          const T gamma)
{
    CUDA_1D_KERNEL_LOOP(index, nthreads)
    {
        int c = index % channels;
        int p = (index / channels) % point_num;
        int n = index / channels / point_num;

        const T* current_point = point + (n * point_num + p) * 2;
        const T* current_offset = offset + (n * point_num + p) * 2;
        const T* current_input = input + (n * channels + c) * height * width;

        const T y = current_point[0] + current_offset[0] * gamma;
        const T x = current_point[1] + current_offset[1] * gamma;

        const T grad_current_output = grad_output[index];

        T* grad_current_input = grad_input + (n * channels + c) * height * width;
        T* grad_current_offset = grad_offset + (n * point_num + p) * 2;

        T w1, w2, w3, w4;
        int x_low, x_high, y_low, y_high;

        bilinear_interpolate_gradient(height,
                                      width,
                                      y,
                                      x,
                                      w1, w2, w3, w4,
                                      y_low, y_high,
                                      x_low, x_high);

        if (x_low >= 0 && x_high >=0 && y_low >= 0 && y_high >= 0)
        {
            atomicAdd(grad_current_input + y_low * width + x_low,
                      grad_current_output * w1);
            atomicAdd(grad_current_input + y_low * width + x_high,
                      grad_current_output * w2);
            atomicAdd(grad_current_input + y_high * width + x_low,
                      grad_current_output * w3);
            atomicAdd(grad_current_input + y_high * width + x_high,
                      grad_current_output * w4);

            T input_00 = current_input[y_low * width + x_low];
            T input_10 = current_input[y_low * width + x_high];
            T input_01 = current_input[y_high * width + x_low];
            T input_11 = current_input[y_high * width + x_high];
            T ogx = gamma * grad_current_output * 
                    (input_11 * (y - y_low) + input_10 * (y_high - y) +
                    input_01 * (y_low - y) + input_00 * (y - y_high));
            T ogy = gamma * grad_current_output * 
                    (input_11 * (x - x_low) + input_01 * (x_high - x) +
                    input_10 * (x_low - x) + input_00 * (x - x_high));
            atomicAdd(grad_current_offset, ogy);
            atomicAdd(grad_current_offset + 1, ogx);
        }
    }
}

#endif  // PROGRESSIVE_SAMPLING_CUDA_KERNEL
