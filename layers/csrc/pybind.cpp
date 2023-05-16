#include "cpp_helper.hpp"

// 这两个函数是用于获取编译器版本和CUDA编译版本的C++函数。 
// `get_compiler_version()` 函数返回编译器的版本信息，例如编译器的名称、版本号等。
// 而 `get_compiling_cuda_version()` 函数返回用于编译CUDA程序的CUDA版本信息，例如CUDA的版本号及其它相关信息。
// 这些函数可以用于在程序运行时获取编译器和CUDA版本的信息，以便在需要时进行诊断和调试。
std::string get_compiler_version();
std::string get_compiling_cuda_version();

// 函数的声明。它指定了函数的名称(progressive_sampling_backward)，参数列表(grad_output, input, point, offset, grad_input, grad_offset, gamma)，
// 以及返回值类型(void)。函数的实际实现可能在另一个文件中定义。该函数的作用和实现细节需要根据具体的上下文和代码来确定。
void progressive_sampling_forward(Tensor input,
                                  Tensor point,
                                  Tensor offset,
                                  Tensor output,
                                  float gamma);

void progressive_sampling_backward(Tensor grad_output,
                                   Tensor input,
                                   Tensor point,
                                   Tensor offset,
                                   Tensor grad_input,
                                   Tensor grad_offset,
                                   float gamma);

// 定义了一个PYBIND11_MODULE，用于将这些CUDA函数绑定到Python中。
// PYBIND11_MODULE()是一个宏，用于将C++函数绑定到Python中。它的作用是将C++函数包装成Python可调用的函数。
// 在这段代码中，PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)将progressive_sampling_forward和progressive_sampling_backward这两个函数绑定到Python中，
// 使得它们可以在Python中被调用。${SELECTED_CODE}中的代码只是函数声明，函数定义可能在其他文件中实现。
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("progressive_sampling_forward", &progressive_sampling_forward,
          "progressive sampling forward",
          py::arg("input"),
          py::arg("point"),
          py::arg("offset"),
          py::arg("output"),
          py::arg("gamma"));
    m.def("progressive_sampling_backward", &progressive_sampling_backward,
          "progressive sampling backward",
          py::arg("grad_output"),
          py::arg("input"),
          py::arg("point"),
          py::arg("offset"),
          py::arg("grad_input"),
          py::arg("grad_offset"),
          py::arg("gamma"));
}
