#include "progressive_sampling_cuda_kernel.cuh"
#include "cuda_helper.hpp"

// 这段代码是一个CUDA内核函数，名为ProgressiveSamplingForwardCUDAKernelLauncher。它有五个输入参数：input、point、offset、output和gamma。
// 其中，input、point和offset是输入张量，output是输出张量，gamma是一个浮点数。这个函数的作用是执行前向传播操作，计算输出张量output的值。
// 在函数内部，首先获取了输出张量output的元素个数output_size，以及输入张量input的通道数channels、高度height和宽度width，以及point张量的点数point_num。
// 然后，使用at::cuda::CUDAGuard和cudaStream_t创建了一个CUDA流stream，并将当前设备设置为输入张量input所在的设备。
// 接下来，使用AT_DISPATCH_FLOATING_TYPES_AND_HALF宏来分发不同类型的输入张量，以便在CUDA内核函数中使用正确的数据类型。
// 在这个宏的作用下，调用了名为progressive_sampling_forward_cuda_kernel的CUDA内核函数，它的输入参数包括输出张量的元素个数output_size
// 以及输入张量input、point、offset和输出张量output的数据指针，以及channels、point_num、height、width和gamma等参数。
// 最后，使用AT_CUDA_CHECK宏检查CUDA内核函数是否执行成功。
void ProgressiveSamplingForwardCUDAKernelLauncher(Tensor input,
                                                  Tensor point,
                                                  Tensor offset,
                                                  Tensor output,
                                                  float gamma)
{
    int output_size = output.numel();
    int channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);
    int point_num = point.size(1);

    at::cuda::CUDAGuard device_guard(input.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(), "progressive_sampling_forward_cuda_kernel", [&] {
            // 这是一个模板函数的调用，用于调用内核函数。`scalar_t`是一个类型参数，表示输入张量的数据类型。
            progressive_sampling_forward_cuda_kernel<scalar_t>
            // 在内核函数调用中，使用了<<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0, stream>>>语法来指定内核函数的执行配置。
            // 其中，GET_BLOCKS(output_size)是一个宏，用于计算内核函数需要的块数，THREADS_PER_BLOCK是一个常量，表示每个块中的线程数，stream是一个CUDA流，用于异步执行内核函数。
                <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0, stream>>>(
                    output_size,
                    input.data_ptr<scalar_t>(),
                    point.data_ptr<scalar_t>(),
                    offset.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    channels,
                    point_num,
                    height,
                    width,
                    gamma
                );
        }
    );

    AT_CUDA_CHECK(cudaGetLastError());
}

void ProgressiveSamplingBackwardCUDAKernelLauncher(Tensor grad_output,
                                                   Tensor input,
                                                   Tensor point,
                                                   Tensor offset,
                                                   Tensor grad_input,
                                                   Tensor grad_offset,
                                                   float gamma)
{
    int output_size = grad_output.numel();
    int channels = grad_input.size(1);
    int height = grad_input.size(2);
    int width = grad_input.size(3);
    int point_num = grad_offset.size(1);

    at::cuda::CUDAGuard device_guard(grad_output.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        grad_output.scalar_type(), "progressive_sampling_backward_cuda_kernel", [&] {

            // 这是一个模板函数的调用，用于调用内核函数。`scalar_t`是一个类型参数，表示输入张量的数据类型。
            progressive_sampling_backward_cuda_kernel<scalar_t>
                <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0, stream>>>(
                    output_size,
                    grad_output.data_ptr<scalar_t>(),
                    input.data_ptr<scalar_t>(),
                    point.data_ptr<scalar_t>(),
                    offset.data_ptr<scalar_t>(),
                    grad_input.data_ptr<scalar_t>(),
                    grad_offset.data_ptr<scalar_t>(),
                    channels,
                    point_num,
                    height,
                    width,
                    gamma
                );
        }
    );

    AT_CUDA_CHECK(cudaGetLastError());
}
