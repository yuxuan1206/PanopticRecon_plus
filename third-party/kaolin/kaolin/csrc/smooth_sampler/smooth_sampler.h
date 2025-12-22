// Copyright (c) 2019,20-21 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//    http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef KAOLIN_SMOOTHSAMPLE_H_
#define KAOLIN_SMOOTHSAMPLE_H_

#include <ATen/ATen.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

namespace kaolin {

torch::Tensor smooth_sampler_forward(torch::Tensor input, torch::Tensor grid,
                                     int64_t padding_mode, bool align_corners, bool apply_smoothstep);

torch::Tensor smooth_sampler_backward(torch::Tensor grad_output, torch::Tensor delta, torch::Tensor corner_val_8); //torch::Tensor corners,torch::Tensor points

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> smooth_sampler_backward_backward(torch::Tensor grad_output, torch::Tensor delta, torch::Tensor corner_val_8, torch::Tensor grad_out_grid); //torch::Tensor corners,torch::Tensor points,

// void launch_smooth_sampler_forward_kernel(
//     const torch::TensorBase &output, const torch::TensorBase &input, const torch::TensorBase &grid,
//     int64_t padding_mode, bool align_corners, bool apply_smoothstep);

// void launch_smooth_sampler_backward_kernel(
//     const torch::TensorBase& grad_input, const torch::TensorBase &grad_grid,
//     const torch::TensorBase& grad_output, const torch::TensorBase& input,
//     const torch::TensorBase& grid, 
//     const torch::TensorBase& points, const torch::TensorBase& corners, const torch::TensorBase& grad_point, const torch::TensorBase& corner_val_8, 
//     int64_t padding_mode, bool align_corners,
//     bool apply_smoothstep, bool input_requires_grad);

// void launch_smooth_sampler_backward_backward_kernel(
//     const torch::TensorBase& grad_input,
//     const torch::TensorBase& grad_grid,
//     const torch::TensorBase& grad_grad_out,
//     const torch::TensorBase& input,
//     const torch::TensorBase& grid,
//     const torch::TensorBase& points, 
//     const torch::TensorBase& corners, const torch::TensorBase& grad_point, const torch::TensorBase& corner_val_8,   
//     const torch::TensorBase& grad_out_input,
//     const torch::TensorBase& grad_out_grid,
//     const torch::TensorBase& grad_output,
//     int64_t padding_mode,
//     const bool align_corners,
//     const bool apply_smoothstep,
//     const bool input_requires_grad);

}  // namespace kaolin

#endif // KAOLIN_METRICS_SIDED_DISTANCE_H_
