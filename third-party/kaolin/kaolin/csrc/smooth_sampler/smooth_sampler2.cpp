/*
  Based on https://github.com/pytorch/pytorch/blob/v1.12.0/aten/src/ATen/native/cuda/GridSampler.cpp
*/

// #include <torch/extension.h>
// #include <c10/cuda/CUDAGuard.h>
#include "smooth_sampler.h"

namespace kaolin {
// #define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
// #define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
// #define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void launch_smooth_sampler_forward_kernel(
    const torch::TensorBase &output, const torch::TensorBase &input, const torch::TensorBase &grid,
    int64_t padding_mode, bool align_corners, bool apply_smoothstep);

void launch_smooth_sampler_backward_kernel(
    const torch::TensorBase &grad_grid,
    const torch::TensorBase& grad_output, 
    const torch::TensorBase& delta, const torch::TensorBase& corner_val_8); //const torch::TensorBase& points, const torch::TensorBase& corners,

void launch_smooth_sampler_backward_backward_kernel(
    const torch::TensorBase& grad_input,
    const torch::TensorBase& grad_grid,
    const torch::TensorBase& grad_grad_out,
    // const torch::TensorBase& input,
    // const torch::TensorBase& grid,
    // const torch::TensorBase& points, 
    // const torch::TensorBase& corners,
    const torch::TensorBase& delta,
    const torch::TensorBase& corner_val_8,   
    // const torch::TensorBase& grad_out_input,
    const torch::TensorBase& grad_out_grid,
    const torch::TensorBase& grad_output);

torch::Tensor smooth_sampler_forward(torch::Tensor input, torch::Tensor grid,
                                     int64_t padding_mode, bool align_corners, bool apply_smoothstep) {
  CHECK_INPUT(input)
  CHECK_INPUT(grid)
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  
  auto in_size = input.sizes();
  auto grid_size = grid.sizes();
  auto output = torch::empty(
      {in_size[0], in_size[1], grid_size[1], grid_size[2], grid_size[3]},
      input.options());
  launch_smooth_sampler_forward_kernel(
      output, input, grid, padding_mode, align_corners, apply_smoothstep);
  return output;
}

torch::Tensor smooth_sampler_backward(torch::Tensor grad_output, torch::Tensor delta, torch::Tensor corner_val_8) { //torch::Tensor corners,torch::Tensor points
  CHECK_INPUT(grad_output)
  // CHECK_INPUT(corners)
  // CHECK_INPUT(points)
  CHECK_INPUT(delta)
  const at::cuda::OptionalCUDAGuard device_guard(device_of(delta));

  // torch::Tensor grad_input = ([&]() {
  //   return torch::zeros_like(corners, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  // })();
  auto grad_grid = torch::empty_like(delta, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  // auto points = torch::empty_like(points, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  // auto corners = torch::empty_like(points, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  // auto grad_point = torch::empty_like(points, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  // auto corner_val_8 = torch::empty_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT); //TODO
  // printf("------------------");
  // std::cout <<delta<<std::endl;
  // printf("\ngrad_output: %f",grad_output);
  launch_smooth_sampler_backward_kernel(
      grad_grid, grad_output,
      delta, corner_val_8); //points, corners,
  return grad_grid.squeeze(0);
  // return std::make_tuple(grad_input, grad_grid);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> smooth_sampler_backward_backward(torch::Tensor grad_output, torch::Tensor delta, torch::Tensor corner_val_8, torch::Tensor grad_out_grid) { //torch::Tensor corners,torch::Tensor points,
  // CHECK_INPUT(grad_out_input)
  CHECK_INPUT(grad_out_grid)
  // CHECK_INPUT(input)
  // CHECK_INPUT(grid)
  CHECK_INPUT(grad_output)
  const at::cuda::OptionalCUDAGuard device_guard(device_of(delta));
  
  auto grad_input = torch::zeros_like(corner_val_8, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  auto grad_grid = torch::empty_like(delta, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  auto grad_grad_out = torch::zeros_like(grad_output, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  // auto points = torch::empty_like(grid, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  // auto corners = torch::empty_like(grid, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  // auto grad_point = torch::empty_like(grid, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  // auto corner_val_8 = torch::empty_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT); //TODO
  launch_smooth_sampler_backward_backward_kernel(grad_input, grad_grid, grad_grad_out,
                                                 delta, corner_val_8,
                                                 grad_out_grid, grad_output); //points, corners
  return std::make_tuple(grad_input, grad_grid, grad_grad_out);
}

// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//   m.def("forward", &smooth_sampler_forward, "Smooth sampler forward (CUDA)");
//   m.def("backward", &smooth_sampler_backward, "Smooth sampler backward (CUDA)");
//   m.def("backward_backward", &smooth_sampler_backward_backward, "Smooth sampler backward backward (CUDA)");
// }
}