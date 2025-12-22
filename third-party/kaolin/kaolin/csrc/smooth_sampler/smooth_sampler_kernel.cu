/*
  Based on https://github.com/pytorch/pytorch/blob/v1.12.0/aten/src/ATen/native/cuda/GridSampler.cu
*/

#include <torch/extension.h>
#include <ATen/cuda/detail/KernelUtils.h>
#include <ATen/native/cuda/KernelUtils.cuh>
#include <ATen/native/cuda/GridSampler.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/IndexUtils.cuh>

#include <stdio.h>
namespace kaolin {
// like at::native::safe_add_3d but without bound check
template<typename scalar_t, typename index_t>
static __forceinline__ __device__
void add_3d(scalar_t *data, int d, int h, int w,
            int sD, int sH, int sW,
            scalar_t delta,
            const index_t NC_offset_inp,
            const index_t memory_span) {
  at::native::fastAtomicAdd(data,
                NC_offset_inp + d * sD + h * sH + w * sW,
                memory_span,
                delta,
                true);
}

__device__ inline float smoothstep(float val) {
	return val * val * (3.0f - 2.0f * val);
}

__device__ inline float smoothstep_derivative(float val) {
	return 6 * val * (1.0f - val);
}

__device__ inline float smoothstep_2nd_derivative(float val) {
	return 6.0f - 12.0f * val;
}

template <typename scalar_t, typename index_t>
C10_LAUNCH_BOUNDS_1(512)
__global__ void smooth_sampler_kernel(
    const index_t nthreads,
    at::cuda::detail::TensorInfo<scalar_t, index_t> input,
    at::cuda::detail::TensorInfo<scalar_t, index_t> grid,
    at::cuda::detail::TensorInfo<scalar_t, index_t> output,
    const at::native::detail::GridSamplerPadding padding_mode,
    bool align_corners,
    bool apply_smoothstep) {
  // printf("----------0-----------\n");
  // printf( "%f, %f, %f, %f, %f\n", input.sizes[0],input.sizes[1],input.sizes[2],input.sizes[3],input.sizes[4] );
  // printf( "%f, %f, %f, %f, %f\n", grid.sizes[0],grid.sizes[1],grid.sizes[2],grid.sizes[3],grid.sizes[4] );  
  index_t C = input.sizes[1];
  index_t inp_D = input.sizes[2];
  index_t inp_H = input.sizes[3];
  index_t inp_W = input.sizes[4];
  index_t out_D = grid.sizes[1]; //N
  index_t out_H = grid.sizes[2];
  index_t out_W = grid.sizes[3];
  index_t inp_sN = input.strides[0];
  index_t inp_sC = input.strides[1];
  index_t inp_sD = input.strides[2];
  index_t inp_sH = input.strides[3];
  index_t inp_sW = input.strides[4];
  index_t grid_sN = grid.strides[0];
  index_t grid_sD = grid.strides[1];
  index_t grid_sH = grid.strides[2];
  index_t grid_sW = grid.strides[3];
  index_t grid_sCoor = grid.strides[4];
  index_t out_sN = output.strides[0];
  index_t out_sC = output.strides[1];
  index_t out_sD = output.strides[2];
  index_t out_sH = output.strides[3];
  index_t out_sW = output.strides[4];

  CUDA_KERNEL_LOOP_TYPE(index, nthreads, index_t) {
    const index_t w = index % out_W;
    const index_t h = (index / out_W) % out_H;
    const index_t d = (index / (out_H * out_W)) % out_D;
    const index_t n = index / (out_D * out_H * out_W);
    const index_t grid_offset = n * grid_sN + d * grid_sD + h * grid_sH + w * grid_sW;

    // get the corresponding input x, y, z co-ordinates from grid
    scalar_t ix = grid.data[grid_offset];
    scalar_t iy = grid.data[grid_offset + grid_sCoor];
    scalar_t iz = grid.data[grid_offset + 2 * grid_sCoor];

    ix = at::native::grid_sampler_compute_source_index(ix, inp_W, padding_mode, align_corners);
    iy = at::native::grid_sampler_compute_source_index(iy, inp_H, padding_mode, align_corners);
    iz = at::native::grid_sampler_compute_source_index(iz, inp_D, padding_mode, align_corners);

    // get corner pixel values from (x, y, z)
    // for 4d, we used north-east-south-west
    // for 5d, we add top-bottom
    index_t _ix = static_cast<index_t>(::floor(ix));
    index_t _iy = static_cast<index_t>(::floor(iy));
    index_t _iz = static_cast<index_t>(::floor(iz));
    index_t ix_ = _ix + 1;
    index_t iy_ = _iy + 1;
    index_t iz_ = _iz + 1;

    scalar_t pos_x_ = ix - _ix;
    scalar_t pos_y_ = iy - _iy;
    scalar_t pos_z_ = iz - _iz;

    if (apply_smoothstep) {
      pos_x_ = smoothstep(pos_x_);
      pos_y_ = smoothstep(pos_y_);
      pos_z_ = smoothstep(pos_z_);
    }

    scalar_t pos_x = 1.0f - pos_x_;
    scalar_t pos_y = 1.0f - pos_y_;
    scalar_t pos_z = 1.0f - pos_z_;

    // get surfaces to each neighbor:
    scalar_t tnw = pos_x  * pos_y  * pos_z;
    scalar_t tne = pos_x_ * pos_y  * pos_z;
    scalar_t tsw = pos_x  * pos_y_ * pos_z;
    scalar_t tse = pos_x_ * pos_y_ * pos_z;
    scalar_t bnw = pos_x  * pos_y  * pos_z_;
    scalar_t bne = pos_x_ * pos_y  * pos_z_;
    scalar_t bsw = pos_x  * pos_y_ * pos_z_;
    scalar_t bse = pos_x_ * pos_y_ * pos_z_;

    auto inp_ptr_NC = input.data + n * inp_sN;
    auto out_ptr_NCDHW = output.data + n * out_sN + d * out_sD + h * out_sH + w * out_sW;
    for (index_t c = 0; c < C; ++c, inp_ptr_NC += inp_sC, out_ptr_NCDHW += out_sC) {
      *out_ptr_NCDHW = static_cast<scalar_t>(0);
      if (at::native::within_bounds_3d(_iz, _iy, _ix, inp_D, inp_H, inp_W)) {
        *out_ptr_NCDHW += inp_ptr_NC[_iz * inp_sD + _iy * inp_sH + _ix * inp_sW] * tnw;
      }
      if (at::native::within_bounds_3d(_iz, _iy, ix_, inp_D, inp_H, inp_W)) {
        *out_ptr_NCDHW += inp_ptr_NC[_iz * inp_sD + _iy * inp_sH + ix_ * inp_sW] * tne;
      }
      if (at::native::within_bounds_3d(_iz, iy_, _ix, inp_D, inp_H, inp_W)) {
        *out_ptr_NCDHW += inp_ptr_NC[_iz * inp_sD + iy_ * inp_sH + _ix * inp_sW] * tsw;
      }
      if (at::native::within_bounds_3d(_iz, iy_, ix_, inp_D, inp_H, inp_W)) {
        *out_ptr_NCDHW += inp_ptr_NC[_iz * inp_sD + iy_ * inp_sH + ix_ * inp_sW] * tse;
      }
      if (at::native::within_bounds_3d(iz_, _iy, _ix, inp_D, inp_H, inp_W)) {
        *out_ptr_NCDHW += inp_ptr_NC[iz_ * inp_sD + _iy * inp_sH + _ix * inp_sW] * bnw;
      }
      if (at::native::within_bounds_3d(iz_, _iy, ix_, inp_D, inp_H, inp_W)) {
        *out_ptr_NCDHW += inp_ptr_NC[iz_ * inp_sD + _iy * inp_sH + ix_ * inp_sW] * bne;
      }
      if (at::native::within_bounds_3d(iz_, iy_, _ix, inp_D, inp_H, inp_W)) {
        *out_ptr_NCDHW += inp_ptr_NC[iz_ * inp_sD + iy_ * inp_sH + _ix * inp_sW] * bsw;
      }
      if (at::native::within_bounds_3d(iz_, iy_, ix_, inp_D, inp_H, inp_W)) {
        *out_ptr_NCDHW += inp_ptr_NC[iz_ * inp_sD + iy_ * inp_sH + ix_ * inp_sW] * bse;
      }
    }
  }
}

//yx
template <typename scalar_t, typename index_t>
C10_LAUNCH_BOUNDS_1(256)
__global__ void smooth_sampler_corner_kernel(
    const index_t nthreads,
    at::cuda::detail::TensorInfo<scalar_t, index_t> grad_output,
    at::cuda::detail::TensorInfo<scalar_t, index_t> input,
    at::cuda::detail::TensorInfo<scalar_t, index_t> grid,
    at::cuda::detail::TensorInfo<scalar_t, index_t> points,   // initialized to empty
    at::cuda::detail::TensorInfo<scalar_t, index_t> corners,   // initialized to empty
    at::cuda::detail::TensorInfo<scalar_t, index_t> grad_point,   // initialized to empty
    at::cuda::detail::TensorInfo<scalar_t, index_t> corner_val_8, // initialized to empty
    at::cuda::detail::TensorInfo<scalar_t, index_t> grad_input,   // initialized to empty
    const at::native::detail::GridSamplerPadding padding_mode,
    bool align_corners,
    const index_t grad_input_memory_span) {
  index_t C = input.sizes[1];
  index_t inp_D = input.sizes[2]; //3
  index_t inp_H = input.sizes[3]; //4
  index_t inp_W = input.sizes[4]; //5
  index_t out_D = grid.sizes[1]; //N
  index_t out_H = grid.sizes[2]; //1
  index_t out_W = grid.sizes[3]; //1
  index_t inp_sN = input.strides[0]; 
  index_t inp_sC = input.strides[1];
  index_t inp_sD = input.strides[2];
  index_t inp_sH = input.strides[3];
  index_t inp_sW = input.strides[4];
  index_t grid_sN = grid.strides[0]; //Nx3
  index_t grid_sD = grid.strides[1]; //3
  index_t grid_sH = grid.strides[2]; 
  index_t grid_sW = grid.strides[3]; 
  index_t grid_sCoor = grid.strides[4]; //1
  index_t gOut_sN = grad_output.strides[0];
  index_t gOut_sC = grad_output.strides[1];
  index_t gOut_sD = grad_output.strides[2];
  index_t gOut_sH = grad_output.strides[3];
  index_t gOut_sW = grad_output.strides[4];
  index_t grid_point_s = points.strides[3];
  index_t grid_corner_s = corners.strides[3];
  index_t grid_grad_s = 3; //grad_point.strides[3];
  int64_t gInp_sN = grad_input.strides[0];
  int64_t gInp_sC = grad_input.strides[1];
  int64_t gInp_sD = grad_input.strides[2];
  int64_t gInp_sH = grad_input.strides[3];
  int64_t gInp_sW = grad_input.strides[4];

  CUDA_KERNEL_LOOP_TYPE(index, nthreads, index_t) {
    const index_t w = index % out_W; //0
    const index_t h = (index / out_W) % out_H; //0
    const index_t d = (index / (out_H * out_W)) % out_D; //针对N的余数d
    const index_t n = index / (out_D * out_H * out_W); //n个N
    const auto grid_offset = n * grid_sN + d * grid_sD + h * grid_sH + w * grid_sW; //n * (Nx3) + d * 3

    // get the corresponding input x, y, z co-ordinates from grid
    scalar_t ix = grid.data[grid_offset];
    scalar_t iy = grid.data[grid_offset + grid_sCoor]; //+1
    scalar_t iz = grid.data[grid_offset + 2 * grid_sCoor]; //+2
    // printf( "\n%f, %f, %f\n", ix,iy,iz);

    // multipliers for gradients on ix, iy, and iz
    scalar_t dL_dix_mult, dL_diy_mult, dL_diz_mult;
    ix = at::native::grid_sampler_compute_source_index_set_grad(ix, inp_W, padding_mode, align_corners, &dL_dix_mult); //5
    iy = at::native::grid_sampler_compute_source_index_set_grad(iy, inp_H, padding_mode, align_corners, &dL_diy_mult); //4
    iz = at::native::grid_sampler_compute_source_index_set_grad(iz, inp_D, padding_mode, align_corners, &dL_diz_mult); //3
    // printf( "***\n%f, %f, %f\n", ix,iy,iz);
    // printf( "-----\n%d, %d, %d, %d, %d\n", ix.sizes[0],input.sizes[1],input.sizes[2],input.sizes[3],input.sizes[4] );
    // get corner pixel values from (x, y, z)
    // for 4d, we used north-east-south-west
    // for 5d, we add top-bottom
    index_t _ix = static_cast<index_t>(::floor(ix)); //left
    index_t _iy = static_cast<index_t>(::floor(iy));
    index_t _iz = static_cast<index_t>(::floor(iz));
    // printf( "&&&\n%f, %f, %f\n", _ix,_iy,_iz);

    index_t ix_ = _ix + 1;
    index_t iy_ = _iy + 1;
    index_t iz_ = _iz + 1;

    //left
    scalar_t pos_x_ = ix - _ix; //[0,1] left
    scalar_t pos_y_ = iy - _iy;
    scalar_t pos_z_ = iz - _iz;

    //right
    scalar_t pos_x = 1.0f - pos_x_; //[0,1] right
    scalar_t pos_y = 1.0f - pos_y_;
    scalar_t pos_z = 1.0f - pos_z_;

    // get surfaces to each neighbor:
    scalar_t tnw = pos_x  * pos_y  * pos_z;
    scalar_t tne = pos_x_ * pos_y  * pos_z;
    scalar_t tsw = pos_x  * pos_y_ * pos_z;
    scalar_t tse = pos_x_ * pos_y_ * pos_z;
    scalar_t bnw = pos_x  * pos_y  * pos_z_;
    scalar_t bne = pos_x_ * pos_y  * pos_z_;
    scalar_t bsw = pos_x  * pos_y_ * pos_z_;
    scalar_t bse = pos_x_ * pos_y_ * pos_z_;

    // scalar_t dL_dix = static_cast<scalar_t>(0), dL_diy = static_cast<scalar_t>(0), dL_diz = static_cast<scalar_t>(0);
    scalar_t *gOut_ptr_NCDHW = grad_output.data + n * gOut_sN + d * gOut_sD + h * gOut_sH + w * gOut_sW;
    // printf( "\n---gOut: %f\n", *gOut_ptr_NCDHW);
    index_t NC_offset_inp;

    NC_offset_inp = n * gInp_sN;

    scalar_t *inp_ptr_NC = input.data + n * inp_sN;
    // // calculate bilinear weighted pixel value and set output pixel
    // for (index_t c = 0; c < C; ++c, gOut_ptr_NCDHW += gOut_sC, NC_offset_inp += gInp_sC, inp_ptr_NC += inp_sC) {
    scalar_t gOut = *gOut_ptr_NCDHW;
    //   printf( "\ngOut: %f\n", gOut);

    at::native::safe_add_3d(grad_input.data, _iz, _iy, _ix, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, tnw * gOut,
                NC_offset_inp, grad_input_memory_span);
    at::native::safe_add_3d(grad_input.data, _iz, _iy, ix_, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, tne * gOut,
                NC_offset_inp, grad_input_memory_span);
    at::native::safe_add_3d(grad_input.data, _iz, iy_, _ix, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, tsw * gOut,
                NC_offset_inp, grad_input_memory_span);
    at::native::safe_add_3d(grad_input.data, _iz, iy_, ix_, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, tse * gOut,
                NC_offset_inp, grad_input_memory_span);
    at::native::safe_add_3d(grad_input.data, iz_, _iy, _ix, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, bnw * gOut,
                NC_offset_inp, grad_input_memory_span);
    at::native::safe_add_3d(grad_input.data, iz_, _iy, ix_, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, bne * gOut,
                NC_offset_inp, grad_input_memory_span);
    at::native::safe_add_3d(grad_input.data, iz_, iy_, _ix, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, bsw * gOut,
                NC_offset_inp, grad_input_memory_span);
    at::native::safe_add_3d(grad_input.data, iz_, iy_, ix_, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, bse * gOut,
                NC_offset_inp, grad_input_memory_span);
      // }
    
    scalar_t *point_ptr = points.data + index * 3;
    point_ptr[0] = ix;
    point_ptr[1] = iy;
    point_ptr[2] = iz;

    scalar_t *corner_ptr = corners.data + index * grid_grad_s;
    corner_ptr[0] = _ix;
    corner_ptr[1] = _iy;
    corner_ptr[2] = _iz;

    scalar_t *grad_ptr = grad_point.data + index * grid_grad_s;
    grad_ptr[0] = dL_dix_mult;
    grad_ptr[1] = dL_diy_mult;
    grad_ptr[2] = dL_diz_mult;

    scalar_t *corner_val_ptr = corner_val_8.data + index * 8;
    // printf("index=%d, C0=%f, C1=%f, C2=%f, C3=%f  ",index,inp_ptr_NC[_iz * inp_sD + _iy * inp_sH + _ix * inp_sW],inp_ptr_NC[_iz * inp_sD + _iy * inp_sH + ix_ * inp_sW],
    // inp_ptr_NC[_iz * inp_sD + iy_ * inp_sH + _ix * inp_sW],inp_ptr_NC[_iz * inp_sD + iy_ * inp_sH + ix_ * inp_sW]);
    // printf("\ncorner_val_ptr=%f",corner_val_ptr[0]);
    corner_val_ptr[0] = inp_ptr_NC[_iz * inp_sD + _iy * inp_sH + _ix * inp_sW];
    corner_val_ptr[1] = inp_ptr_NC[_iz * inp_sD + _iy * inp_sH + ix_ * inp_sW];
    corner_val_ptr[2] = inp_ptr_NC[_iz * inp_sD + iy_ * inp_sH + _ix * inp_sW];
    corner_val_ptr[3] = inp_ptr_NC[_iz * inp_sD + iy_ * inp_sH + ix_ * inp_sW];
    corner_val_ptr[4] = inp_ptr_NC[iz_ * inp_sD + _iy * inp_sH + _ix * inp_sW];
    corner_val_ptr[5] = inp_ptr_NC[iz_ * inp_sD + _iy * inp_sH + ix_ * inp_sW];
    corner_val_ptr[6] = inp_ptr_NC[iz_ * inp_sD + iy_ * inp_sH + _ix * inp_sW];
    corner_val_ptr[7] = inp_ptr_NC[iz_ * inp_sD + iy_ * inp_sH + ix_ * inp_sW];
  }
  // printf("\n=========\n");
}

//yx
template <typename scalar_t, typename index_t>
C10_LAUNCH_BOUNDS_1(256)
__global__ void smooth_sampler_backward_kernel2(
    const index_t nthreads,
    at::cuda::detail::TensorInfo<scalar_t, index_t> grad_output,
    // at::cuda::detail::TensorInfo<scalar_t, index_t> corners,   //input [1,N,3]
    // at::cuda::detail::TensorInfo<scalar_t, index_t> points, //coords [1,N,3]
    at::cuda::detail::TensorInfo<scalar_t, index_t> delta, // [1,N,3]
    // at::cuda::detail::TensorInfo<scalar_t, index_t> grad_point, 
    at::cuda::detail::TensorInfo<scalar_t, index_t> corner_val_8, 
    // at::cuda::detail::TensorInfo<scalar_t, index_t> grad_input,  // initialized to zeros (or unused if input_requires_grad is false)
    at::cuda::detail::TensorInfo<scalar_t, index_t> grad_grid) { // initialized to empty
  
  index_t out_D = delta.sizes[1]; //N
  index_t grid_sN = delta.strides[0]; //Nx3
  index_t grid_sD = delta.strides[1]; //3
  index_t grid_sCoor = delta.strides[2]; //1

  index_t gOut_D = grad_output.sizes[1]; //N
  index_t gOut_H = grad_output.sizes[2]; //6 一个块插值的值的个数
  index_t gOut_sN = grad_output.strides[0]; //N*6
  index_t gOut_sD = grad_output.strides[1]; //6

  index_t cVal_sN = corner_val_8.strides[0]; //N*6*8
  index_t cVal_sD = corner_val_8.strides[1]; //6*8
  index_t cVal_sH = corner_val_8.strides[2]; //8
 
  // index_t gGrid_sW = grad_grid.strides[1];

  CUDA_KERNEL_LOOP_TYPE(index, nthreads, index_t) {
    const index_t d = index % out_D; //index % N 
    const index_t n = index / out_D; //index / N 
    const auto grid_offset = n * grid_sN + d * grid_sD; //n * (Nx3) + d * 3
    // printf( "index:%d, %d, %d, %d\n", index,d,n,grid_offset);
    // const auto grid_offset = index * 3;
    
    scalar_t pos_x_ = delta.data[grid_offset]; //[0,1] right 
    scalar_t pos_y_ = delta.data[grid_offset+1];
    scalar_t pos_z_ = delta.data[grid_offset+2];
    // if (index==0){
    //   printf( "delta: %.4f, %.4f, %.4f\n", pos_x_,pos_y_,pos_z_);
    // }
  
    float pos_x_derivative = 1.0f;
    float pos_y_derivative = 1.0f;
    float pos_z_derivative = 1.0f;

    //right
    scalar_t pos_x = 1.0f - pos_x_; //[0,1] left
    scalar_t pos_y = 1.0f - pos_y_;
    scalar_t pos_z = 1.0f - pos_z_;
    // if (index==0){
    //   printf( "1-delta: %.4f, %.4f, %.4f\n", pos_x,pos_y,pos_z);
    // }

    scalar_t dL_dix = static_cast<scalar_t>(0), dL_diy = static_cast<scalar_t>(0), dL_diz = static_cast<scalar_t>(0);
    // const auto grid_offset = n_ * gOut_sN + d_ * gOut_D + h_; //n * (Nx3) + d * 3
    // scalar_t *gOut_ptr_NCDHW = grad_output.data + n * gOut_sN + d * gOut_sD;//n * gOut_sN + d * gOut_sD;
    // scalar_t gOut = *gOut_ptr_NCDHW;
    // printf( "n=%d, d=%d ", n,d);
    // printf( "gOut:%f  ", 10000* *grad_output.data);

    // calculate grad_grid
    const auto gOut_offset = n * gOut_sN + d * gOut_sD;
    const auto val_offset = n * cVal_sN + d * cVal_sD; // n * (Nx6x8) + d * (6x8) 第几块的查询点
  // if (at::native::within_bounds_3d(_iz, _iy, _ix, inp_D, inp_H, inp_W)) {
    // scalar_t tnw_val = inp_ptr_NC[_iz * inp_sD + _iy * inp_sH + _ix * inp_sW];
    for(size_t i=0;i<gOut_H;i++){
      scalar_t gOut = grad_output.data[gOut_offset + i];
      
      scalar_t tnw_val = corner_val_8.data[val_offset + i * cVal_sH]; //1-x 1-y 1-z
      dL_dix -= tnw_val * (pos_y) * (pos_z) * gOut; //1-x 1-y 1-z
      dL_diy -= tnw_val * (pos_x) * (pos_z) * gOut;
      dL_diz -= tnw_val * (pos_x) * (pos_y) * gOut;
      // if (index==0){
      //   printf( "i = %d, grad = %.4f (idx:%d), corner = %.4f (idx:%d), dL_dix = %.4f, cVal_sH=%d\n", i,gOut,gOut_offset+i,tnw_val,val_offset+i*cVal_sH,dL_dix,cVal_sH);
      // }
      // printf("\n====0=====\n");
    // }
    // if (at::native::within_bounds_3d(_iz, _iy, ix_, inp_D, inp_H, inp_W)) {
      // scalar_t tne_val = inp_ptr_NC[_iz * inp_sD + _iy * inp_sH + ix_ * inp_sW];
      scalar_t tne_val = corner_val_8.data[val_offset + i * cVal_sH+1]; //1-x 1-y z
      dL_dix -= tne_val * (pos_y) * (pos_z_) * gOut; //1-x 1-y z
      dL_diy -= tne_val * (pos_x) * (pos_z_) * gOut;
      dL_diz += tne_val * (pos_x) * (pos_y) * gOut;
      // printf("\n====1=====\n");
    // }
    // if (at::native::within_bounds_3d(_iz, iy_, _ix, inp_D, inp_H, inp_W)) {
      // scalar_t tsw_val = inp_ptr_NC[_iz * inp_sD + iy_ * inp_sH + _ix * inp_sW];
      scalar_t tsw_val = corner_val_8.data[val_offset + i * cVal_sH+2]; //1-x y 1-z
      dL_dix -= tsw_val * (pos_y_) * (pos_z) * gOut; //1-x y 1-z
      dL_diy += tsw_val * (pos_x) * (pos_z) * gOut;
      dL_diz -= tsw_val * (pos_x) * (pos_y_) * gOut;
      // printf("\n====2=====\n");
    // }
    // if (at::native::within_bounds_3d(_iz, iy_, ix_, inp_D, inp_H, inp_W)) {
      // scalar_t tse_val = inp_ptr_NC[_iz * inp_sD + iy_ * inp_sH + ix_ * inp_sW];
      scalar_t tse_val = corner_val_8.data[val_offset + i * cVal_sH+3]; //1-x y z
      dL_dix -= tse_val * (pos_y_) * (pos_z_) * gOut; //1-x y z
      dL_diy += tse_val * (pos_x) * (pos_z_) * gOut;
      dL_diz += tse_val * (pos_x) * (pos_y_) * gOut;
      // printf("\n====3=====\n");
    // }
    // if (at::native::within_bounds_3d(iz_, _iy, _ix, inp_D, inp_H, inp_W)) {
      // scalar_t bnw_val = inp_ptr_NC[iz_ * inp_sD + _iy * inp_sH + _ix * inp_sW];
      scalar_t bnw_val = corner_val_8.data[val_offset + i * cVal_sH+4]; //x 1-y 1-z
      dL_dix += bnw_val * (pos_y) * (pos_z) * gOut; //x 1-y 1-z
      dL_diy -= bnw_val * (pos_x_) * (pos_z) * gOut;
      dL_diz -= bnw_val * (pos_x_) * (pos_y) * gOut;
      // printf("\n====4=====\n");
    // }
    // if (at::native::within_bounds_3d(iz_, _iy, ix_, inp_D, inp_H, inp_W)) {
      // scalar_t bne_val = inp_ptr_NC[iz_ * inp_sD + _iy * inp_sH + ix_ * inp_sW];
      scalar_t bne_val = corner_val_8.data[val_offset + i * cVal_sH+5]; //x 1-y z
      dL_dix += bne_val * (pos_y) * (pos_z_) * gOut; //x 1-y z
      dL_diy -= bne_val * (pos_x_) * (pos_z_) * gOut;
      dL_diz += bne_val * (pos_x_) * (pos_y) * gOut;
      // printf("\n====5=====\n");
    // }
    // if (at::native::within_bounds_3d(iz_, iy_, _ix, inp_D, inp_H, inp_W)) {
      // scalar_t bsw_val = inp_ptr_NC[iz_ * inp_sD + iy_ * inp_sH + _ix * inp_sW];
      scalar_t bsw_val = corner_val_8.data[val_offset + i * cVal_sH+6]; //x y 1-z
      dL_dix += bsw_val * (pos_y_) * (pos_z) * gOut; //x y 1-z
      dL_diy += bsw_val * (pos_x_) * (pos_z) * gOut;
      dL_diz -= bsw_val * (pos_x_) * (pos_y_) * gOut;
      // printf("\n====6=====\n");
    // }
    // if (at::native::within_bounds_3d(iz_, iy_, ix_, inp_D, inp_H, inp_W)) {
      // scalar_t bse_val = inp_ptr_NC[iz_ * inp_sD + iy_ * inp_sH + ix_ * inp_sW];
      scalar_t bse_val = corner_val_8.data[val_offset + i * cVal_sH+7]; //x y z
      dL_dix += bse_val * (pos_y_) * (pos_z_) * gOut; //x y z
      dL_diy += bse_val * (pos_x_) * (pos_z_) * gOut;
      dL_diz += bse_val * (pos_x_) * (pos_y_) * gOut;
      // printf("\n====0=====\n");
    // }
    }
    
  // }
    // assuming grad_grid is contiguous
    // thus we can
    //   1. use index with gGrid_sW to directly compute gGrid_ptr_NDHW
    //   2. directly assign to gGrid_ptr_NDHW[0], gGrid_ptr_NDHW[1], gGrid_ptr_NDHW[2]
    scalar_t *gGrid_ptr_NDHW = grad_grid.data + index * 3;
    gGrid_ptr_NDHW[0] = dL_dix * pos_x_derivative; //dL_dix_mult * dL_dix * pos_x_derivative;
    gGrid_ptr_NDHW[1] = dL_diy * pos_y_derivative; //dL_diy_mult * dL_diy * pos_y_derivative;
    gGrid_ptr_NDHW[2] = dL_diz * pos_z_derivative; //dL_diz_mult * dL_diz * pos_z_derivative;
    // printf( "\ngrad_coord: %f, %f, %f\n", 10000*gGrid_ptr_NDHW[0],10000*gGrid_ptr_NDHW[1],10000*gGrid_ptr_NDHW[2]);
  }
}

template <typename scalar_t, typename index_t>
C10_LAUNCH_BOUNDS_1(256)
__global__ void smooth_sampler_backward_backward_kernel2(
    const index_t nthreads,
    at::cuda::detail::TensorInfo<scalar_t, index_t> grad_input, // initialized to empty
    at::cuda::detail::TensorInfo<scalar_t, index_t> grad_grid, // initialized to zeros
    at::cuda::detail::TensorInfo<scalar_t, index_t> grad_grad_out, // initialized to zeros
    // at::cuda::detail::TensorInfo<scalar_t, index_t> corners,
    // at::cuda::detail::TensorInfo<scalar_t, index_t> points, //coords
    at::cuda::detail::TensorInfo<scalar_t, index_t> delta,
    // at::cuda::detail::TensorInfo<scalar_t, index_t> grad_point, 
    at::cuda::detail::TensorInfo<scalar_t, index_t> corner_val_8, 
    // at::cuda::detail::TensorInfo<scalar_t, index_t> grad_out_input,
    at::cuda::detail::TensorInfo<scalar_t, index_t> grad_out_grid,
    at::cuda::detail::TensorInfo<scalar_t, index_t> grad_output,
    // bool input_requires_grad,
    const index_t grad_input_memory_span,
    const index_t grad_grad_out_memory_span) {
  // printf("11--");
  index_t out_D = delta.sizes[1]; //N
  index_t out_H = delta.sizes[2]; //3
  // index_t out_W = points.sizes[3];
  index_t grid_sN = delta.strides[0]; //Nx3
  index_t grid_sD = delta.strides[1]; //3
  index_t grid_sCoor = delta.strides[2]; //1

  index_t gGrid_sW = grad_grid.strides[1]; //3

  index_t gOut_D = grad_output.sizes[1]; //N
  index_t gOut_H = grad_output.sizes[2]; //6
  index_t gOut_sN = grad_output.strides[0]; //N*6
  index_t gOut_sD = grad_output.strides[1]; //6

  index_t cVal_sN = corner_val_8.strides[0]; //N*6*8
  index_t cVal_sD = corner_val_8.strides[1]; //6*8
  index_t cVal_sH = corner_val_8.strides[2]; //8

  index_t gOutGrid_sW = grad_out_grid.strides[0]; //3

  index_t gOutInput_sN = 0;
  index_t gOutInput_sC = 0;

  // if (input_requires_grad) {
  // gOutInput_sN = grad_out_input.strides[0]; //N*6*8
  // gOutInput_sC = grad_out_input.strides[1]; //6*8
  // }

  index_t gInp_sN = grad_input.strides[0]; //N*6*8
  index_t gInp_sC = grad_input.strides[1]; //6*8
  index_t gInp_sD = grad_input.strides[2]; //8
  index_t gInp_sH = grad_input.strides[3]; //1
  // index_t gInp_sW = grad_input.strides[4];
  // printf("22--");
  CUDA_KERNEL_LOOP_TYPE(index, nthreads, index_t) {
    const index_t d = index % out_D;
    const index_t n = index / out_D;
    const auto grid_offset = n * grid_sN + d * grid_sD;
    // printf( "%f, %f, %f, %f\n", w,h,d,n );
    // printf( "%f, %f, %f, %f\n", grid_sN,grid_sD,grid_sH,grid_sW );
    // if (index==0){
    //   // printf( "grad_out_grid.size: %d, %d，%d\n", grad_out_grid.sizes[0],grad_out_grid.sizes[1],grad_out_grid.sizes[2]);
    //   printf( "corner_val_8.size: %d, %d，%d, %d\n", corner_val_8.sizes[0],corner_val_8.sizes[1],corner_val_8.sizes[2],corner_val_8.sizes[3]);
    //   printf( "corner_val_8.strides: %d, %d，%d\n", cVal_sN,cVal_sD,cVal_sH);
    // }

    scalar_t pos_x_ = delta.data[grid_offset]; //x right 
    scalar_t pos_y_ = delta.data[grid_offset+1]; //y
    scalar_t pos_z_ = delta.data[grid_offset+2]; //z

    scalar_t pos_x_derivative_ = 1.0f; //dL_dix_mult;
    scalar_t pos_y_derivative_ = 1.0f; //dL_diy_mult;
    scalar_t pos_z_derivative_ = 1.0f; //dL_diz_mult;

    scalar_t pos_x_2nd_derivative_ = 0.0f;
    scalar_t pos_y_2nd_derivative_ = 0.0f;
    scalar_t pos_z_2nd_derivative_ = 0.0f;

    scalar_t pos_x_2nd_derivative = 0.0f;
    scalar_t pos_y_2nd_derivative = 0.0f;
    scalar_t pos_z_2nd_derivative = 0.0f;

    scalar_t pos_x = 1.0f - pos_x_; //1-x
    scalar_t pos_y = 1.0f - pos_y_; //1-y
    scalar_t pos_z = 1.0f - pos_z_; //1-z

    scalar_t pos_x_derivative = -pos_x_derivative_;
    scalar_t pos_y_derivative = -pos_y_derivative_;
    scalar_t pos_z_derivative = -pos_z_derivative_;

    // index_t index_corners[2][3] = {{_ix, _iy, _iz},
    //                                {ix_, iy_, iz_}};
    scalar_t pos_corners[2][9] = {{pos_x, pos_y, pos_z, //(1-x) (1-y) (1-z)
                                   pos_x_derivative, pos_y_derivative, pos_z_derivative, //d(1-x)/dx_old    -grad_out 1_degree
                                   pos_x_2nd_derivative, pos_y_2nd_derivative, pos_z_2nd_derivative}, //2_degree
                                  {pos_x_, pos_y_, pos_z_, //x y z
                                   pos_x_derivative_, pos_y_derivative_, pos_z_derivative_, //dx/dx_old     grad_out
                                   pos_x_2nd_derivative_, pos_y_2nd_derivative_, pos_z_2nd_derivative_}}; //2_degree
    scalar_t surface_coefficients[8] = {};
    scalar_t out_derivatives[8][12] = {};
    // printf("33--");
    #pragma unroll
    for (int shift = 0; shift < 8; shift++) { 
      // printf("shift=%d",shift);
      int pz = (shift >> 0) & 1; //shift & 1
      int py = (shift >> 1) & 1; //shift/2 & 1
      int px = (shift >> 2) & 1; //shift/4 & 1
      // 000
      // 100
      // 010
      // 110
      // 001
      // 101
      // 011
      // 111
  
      // printf("shift=%d, px=%d, py=%d, pz=%d \n",shift,px,py,pz);

      // surface_coefficients[shift] = pos_corners[px][0] * pos_corners[py][1] * pos_corners[pz][2]; //w(X): ()()(), ()()z,  ...

      out_derivatives[shift][0] = pos_corners[py][1] * pos_corners[pz][2] * pos_corners[px][3]; // dOut_dx / surf_weight (1 degree grad) -(1-y)(1-z),-(1-y)z...
      out_derivatives[shift][1] = pos_corners[py][1] * pos_corners[pz][2] * pos_corners[px][6]; // d2Out_dx2 / surf_weight  0 0 ...
      out_derivatives[shift][2] = pos_corners[py][4] * pos_corners[pz][2] * pos_corners[px][3]; // d2Out_dxdy / surf_weight  (1-z) z ...
      out_derivatives[shift][3] = pos_corners[py][1] * pos_corners[pz][5] * pos_corners[px][3]; // d2Out_dxdz / surf_weight  (1-y) -(1-y)...

      out_derivatives[shift][4] = pos_corners[px][0] * pos_corners[pz][2] * pos_corners[py][4]; // dOut_dy / surf_weight
      out_derivatives[shift][5] = pos_corners[px][0] * pos_corners[pz][2] * pos_corners[py][7]; // d2Out_dy2 / surf_weight  0 0 ...
      out_derivatives[shift][6] = pos_corners[px][3] * pos_corners[pz][2] * pos_corners[py][4]; // d2Out_dydx / surf_weight
      out_derivatives[shift][7] = pos_corners[px][0] * pos_corners[pz][5] * pos_corners[py][4]; // d2Out_dydz / surf_weight

      out_derivatives[shift][8] = pos_corners[px][0] * pos_corners[py][1] * pos_corners[pz][5]; // dOut_dz / surf_weight
      out_derivatives[shift][9] = pos_corners[px][0] * pos_corners[py][1] * pos_corners[pz][8]; // d2Out_dz2 / surf_weight 0 0 ...
      out_derivatives[shift][10] = pos_corners[px][3] * pos_corners[py][1] * pos_corners[pz][5]; // d2Out_dzdx / surf_weight
      out_derivatives[shift][11] = pos_corners[px][0] * pos_corners[py][4] * pos_corners[pz][5]; // d2Out_dzdy / surf_weight
      // if (index==0){
      //   printf( "shift: %d, px: %d, py: %d, pz: %d\n", shift,px,py,pz);
      // }
    }

    scalar_t d2L_dix2 = static_cast<scalar_t>(0), d2L_diy2 = static_cast<scalar_t>(0), d2L_diz2 = static_cast<scalar_t>(0);
    index_t offset_out_DHW = d * gOut_sD;
    // scalar_t *gOut_ptr_NCDHW = grad_output.data + n * gOut_sN + offset_out_DHW;
    index_t NC_offset_inp = n * gInp_sN;
    index_t NC_offset_out = n * gOut_sN; //n*N*6
    // scalar_t *inp_ptr_NC = input.data + n * inp_sN;

    scalar_t *gOutInput_ptr_NC = NULL;

    // if (input_requires_grad) {
    //   gOutInput_ptr_NC = grad_out_input.data + n * gOutInput_sN;
    // }

    scalar_t *gOutGrid_ptr_NDHW = grad_out_grid.data + index * gOutGrid_sW;
    scalar_t *gGrid_ptr_NDHW = grad_grid.data + index * gGrid_sW;
    const auto gOut_offset = n * gOut_sN + d * gOut_sD;
    const auto val_offset = n * cVal_sN + d * cVal_sD;
    // printf("===========================");
    // for (index_t c = 0; c < C; ++c, gOut_ptr_NCDHW += gOut_sC, inp_ptr_NC += inp_sC, gOutInput_ptr_NC += gOutInput_sC, NC_offset_inp += gInp_sC, NC_offset_out += gOut_sC) {
    for(size_t i=0;i<gOut_H;i++){
      scalar_t gOut = grad_output.data[gOut_offset + i];
      // scalar_t gOut = *gOut_ptr_NCDHW;
      // const auto val_offset = n * out_D * 8 + d * 8;

      #pragma unroll
      for (int shift = 0; shift < 8; shift++) {

        // Slightly unprecise naming: in fact these are divided by surf_weight.
        scalar_t dOut_dx = out_derivatives[shift][0]; // E.g. variable "dOut_dx" is mathematically "dOut/dx * 1/surf_weight"
        scalar_t d2Out_dx2 = out_derivatives[shift][1]; 
        scalar_t d2Out_dxdy = out_derivatives[shift][2];
        scalar_t d2Out_dxdz = out_derivatives[shift][3];
        scalar_t dOut_dy = out_derivatives[shift][4];
        scalar_t d2Out_dy2 = out_derivatives[shift][5];
        scalar_t d2Out_dydx = out_derivatives[shift][6];
        scalar_t d2Out_dydz = out_derivatives[shift][7];
        scalar_t dOut_dz = out_derivatives[shift][8];
        scalar_t d2Out_dz2 = out_derivatives[shift][9];
        scalar_t d2Out_dzdx = out_derivatives[shift][10];
        scalar_t d2Out_dzdy = out_derivatives[shift][11];
        scalar_t surface_coeff = surface_coefficients[shift];

        // if (at::native::within_bounds_3d(iz, iy, ix, inp_D, inp_H, inp_W)) {
          // index_t inp_el = iz * inp_sD + iy * inp_sH + ix * inp_sW;
          // scalar_t surf_weight = inp_ptr_NC[inp_el];
        // scalar_t surf_weight = corner_val_8.data[val_offset + ix + iy<<1 + iz<<2];
        scalar_t surf_weight = corner_val_8.data[val_offset + i * cVal_sH + shift]; // corner value
        // index_t idx= i * cVal_sH + shift;
        // if (index==0){
        //   // printf( "corner_val_8.strides: %d\n", cVal_sH);
        //   printf( "idx: %d, surf_weight: %f\n", idx,surf_weight);
        //   // printf( "shift: %d, val_offset: %d, i: %d\n", shift,val_offset,i);
        //   // printf( "surf_weight: %f\n", surf_weight);
        // }


        scalar_t dL_dx = gOut * dOut_dx;
        scalar_t dL_dy = gOut * dOut_dy;
        scalar_t dL_dz = gOut * dOut_dz;

        scalar_t gOutGrid_x = gOutGrid_ptr_NDHW[0];
        scalar_t gOutGrid_y = gOutGrid_ptr_NDHW[1];
        scalar_t gOutGrid_z = gOutGrid_ptr_NDHW[2];
        // if (index==0){
        //     // printf( "corner_val_8.strides: %d\n", cVal_sH);
        //     printf( "gOutGrid_x: %f, gOutGrid_x: %f, gOutGrid_x: %f\n", gOutGrid_x,gOutGrid_y,gOutGrid_z);
        //     // printf( "shift: %d, val_offset: %d, i: %d\n", shift,val_offset,i);
        //     // printf( "surf_weight: %f\n", surf_weight);
        //   }
  

        scalar_t grad_grad_out_delta = surf_weight * (dOut_dx * gOutGrid_x
                                                      + dOut_dy * gOutGrid_y
                                                      + dOut_dz * gOutGrid_z);

        // if (gOutInput_ptr_NC != NULL) {
        //   index_t gOutInput_el = iz * gOutInput_sD + iy * gOutInput_sH + ix * gOutInput_sW;
        //   scalar_t gOutInput = gOutInput_ptr_NC[gOutInput_el];
        //   grad_grad_out_delta += gOutInput * surface_coeff;
        //   d2L_dix2 += dL_dx * gOutInput;
        //   d2L_diy2 += dL_dy * gOutInput;
        //   d2L_diz2 += dL_dz * gOutInput;
        // }

        at::native::fastAtomicAdd(grad_grad_out.data,
                                  NC_offset_out + offset_out_DHW,
                                  grad_grad_out_memory_span,
                                  grad_grad_out_delta,
                                  true);

        d2L_dix2 += surf_weight * gOut * (d2Out_dx2 * gOutGrid_x
                                          + d2Out_dxdy * gOutGrid_y
                                          + d2Out_dxdz * gOutGrid_z);
        d2L_diy2 += surf_weight * gOut * (d2Out_dydx * gOutGrid_x
                                          + d2Out_dy2 * gOutGrid_y
                                          + d2Out_dydz * gOutGrid_z);
        d2L_diz2 += surf_weight * gOut * (d2Out_dzdx * gOutGrid_x
                                          + d2Out_dzdy * gOutGrid_y
                                          + d2Out_dz2 * gOutGrid_z);
        
        // add_3d(grad_input.data, iz, iy, ix, gInp_sD, gInp_sH, gInp_sW,
        //       dL_dx * gOutGrid_x + dL_dy * gOutGrid_y + dL_dz * gOutGrid_z,
        //       NC_offset_inp, grad_input_memory_span);
        index_t idx = val_offset + i * cVal_sH + shift;
        at::native::fastAtomicAdd(grad_input.data,
                                  idx,
                                  grad_input_memory_span,
                                  dL_dx * gOutGrid_x + dL_dy * gOutGrid_y + dL_dz * gOutGrid_z,
                                  true);
        }
      }

    gGrid_ptr_NDHW[0] = d2L_dix2; //d2f/d2x
    gGrid_ptr_NDHW[1] = d2L_diy2;
    gGrid_ptr_NDHW[2] = d2L_diz2;
  }
}

template <typename scalar_t, typename index_t>
C10_LAUNCH_BOUNDS_1(256)
__global__ void smooth_sampler_backward_kernel(
    const index_t nthreads,
    at::cuda::detail::TensorInfo<scalar_t, index_t> grad_output,
    at::cuda::detail::TensorInfo<scalar_t, index_t> input,
    at::cuda::detail::TensorInfo<scalar_t, index_t> grid,
    at::cuda::detail::TensorInfo<scalar_t, index_t> grad_input,  // initialized to zeros (or unused if input_requires_grad is false)
    at::cuda::detail::TensorInfo<scalar_t, index_t> grad_grid,   // initialized to empty
    const at::native::detail::GridSamplerPadding padding_mode,
    bool align_corners,
    bool apply_smoothstep,
    const index_t grad_input_memory_span,
    const bool input_requires_grad) {
  printf("----------1-----------\n");
  printf( "%d, %d, %d, %d, %d\n", input.sizes[0],input.sizes[1],input.sizes[2],input.sizes[3],input.sizes[4] );
  // printf( "%d, %d, %d, %d, %d\n", grid.sizes[0],grid.sizes[1],grid.sizes[2],grid.sizes[3],grid.sizes[4] );
  // printf( "%d, %d, %d, %d, %d\n", grad_output.sizes[0],grad_output.sizes[1],grad_output.sizes[2],grad_output.sizes[3],grad_output.sizes[4] );
  index_t C = input.sizes[1]; 
  printf("C = %d\n",C);
  index_t inp_D = input.sizes[2]; //3
  index_t inp_H = input.sizes[3]; //4
  index_t inp_W = input.sizes[4]; //5
  index_t out_D = grid.sizes[1]; //N
  index_t out_H = grid.sizes[2]; //1
  index_t out_W = grid.sizes[3]; //1
  index_t inp_sN = input.strides[0]; 
  index_t inp_sC = input.strides[1];
  index_t inp_sD = input.strides[2];
  index_t inp_sH = input.strides[3];
  index_t inp_sW = input.strides[4];
  index_t grid_sN = grid.strides[0]; //Nx3
  index_t grid_sD = grid.strides[1]; //3
  index_t grid_sH = grid.strides[2]; 
  index_t grid_sW = grid.strides[3]; 
  index_t grid_sCoor = grid.strides[4]; //1
  index_t gOut_sN = grad_output.strides[0];
  index_t gOut_sC = grad_output.strides[1];
  index_t gOut_sD = grad_output.strides[2];
  index_t gOut_sH = grad_output.strides[3];
  index_t gOut_sW = grad_output.strides[4];
  // gInp_* (and NC_offset_inp below) are not really needed if input_requires_grad is false.
  int64_t gInp_sN = 0;
  int64_t gInp_sC = 0;
  int64_t gInp_sD = 0;
  int64_t gInp_sH = 0;
  int64_t gInp_sW = 0;
  if (input_requires_grad) {
    gInp_sN = grad_input.strides[0];
    gInp_sC = grad_input.strides[1];
    gInp_sD = grad_input.strides[2];
    gInp_sH = grad_input.strides[3];
    gInp_sW = grad_input.strides[4];
  }
  index_t gGrid_sW = grad_grid.strides[3];

  CUDA_KERNEL_LOOP_TYPE(index, nthreads, index_t) {
    const index_t w = index % out_W; //0
    const index_t h = (index / out_W) % out_H; //0
    const index_t d = (index / (out_H * out_W)) % out_D; //针对N的余数d
    const index_t n = index / (out_D * out_H * out_W); //n个N
    const auto grid_offset = n * grid_sN + d * grid_sD + h * grid_sH + w * grid_sW; //n * (Nx3) + d * 3
    // printf( "-----\n%d, %d, %d, %d, %d\n", input.strides[0],input.strides[1],input.strides[2],input.strides[3],input.strides[4] );
    // printf( "=====\n%d, %d, %d, %d, %d\n", grid.strides[0],grid.strides[1],grid.strides[2],grid.strides[3],grid.strides[4] );

    // get the corresponding input x, y, z co-ordinates from grid
    scalar_t ix = grid.data[grid_offset];
    scalar_t iy = grid.data[grid_offset + grid_sCoor]; //+1
    scalar_t iz = grid.data[grid_offset + 2 * grid_sCoor]; //+2
    // printf( "\n%f, %f, %f\n", ix,iy,iz);

    // multipliers for gradients on ix, iy, and iz
    scalar_t dL_dix_mult, dL_diy_mult, dL_diz_mult;
    ix = at::native::grid_sampler_compute_source_index_set_grad(ix, inp_W, padding_mode, align_corners, &dL_dix_mult); //5
    iy = at::native::grid_sampler_compute_source_index_set_grad(iy, inp_H, padding_mode, align_corners, &dL_diy_mult); //4
    iz = at::native::grid_sampler_compute_source_index_set_grad(iz, inp_D, padding_mode, align_corners, &dL_diz_mult); //3
    // printf( "***\n%f, %f, %f\n", ix,iy,iz);
    // printf( "-----\n%d, %d, %d, %d, %d\n", ix.sizes[0],input.sizes[1],input.sizes[2],input.sizes[3],input.sizes[4] );
    // get corner pixel values from (x, y, z)
    // for 4d, we used north-east-south-west
    // for 5d, we add top-bottom
    index_t _ix = static_cast<index_t>(::floor(ix)); //left
    index_t _iy = static_cast<index_t>(::floor(iy));
    index_t _iz = static_cast<index_t>(::floor(iz));
    printf( "\ncorners: %d, %d, %d\n", _ix,_iy,_iz);
    index_t ix_ = _ix + 1; //right
    index_t iy_ = _iy + 1;
    index_t iz_ = _iz + 1;

    //left
    scalar_t pos_x_ = ix - _ix; //[0,1] left
    scalar_t pos_y_ = iy - _iy;
    scalar_t pos_z_ = iz - _iz;

    float pos_x_derivative = 1.0f;
    float pos_y_derivative = 1.0f;
    float pos_z_derivative = 1.0f;

    if (apply_smoothstep) {
      pos_x_derivative = smoothstep_derivative(pos_x_);
      pos_y_derivative = smoothstep_derivative(pos_y_);
      pos_z_derivative = smoothstep_derivative(pos_z_);
      pos_x_ = smoothstep(pos_x_);
      pos_y_ = smoothstep(pos_y_);
      pos_z_ = smoothstep(pos_z_);
    }
    //right
    scalar_t pos_x = 1.0f - pos_x_; //[0,1] right
    scalar_t pos_y = 1.0f - pos_y_;
    scalar_t pos_z = 1.0f - pos_z_;

    // get surfaces to each neighbor:
    scalar_t tnw = pos_x  * pos_y  * pos_z;
    scalar_t tne = pos_x_ * pos_y  * pos_z;
    scalar_t tsw = pos_x  * pos_y_ * pos_z;
    scalar_t tse = pos_x_ * pos_y_ * pos_z;
    scalar_t bnw = pos_x  * pos_y  * pos_z_;
    scalar_t bne = pos_x_ * pos_y  * pos_z_;
    scalar_t bsw = pos_x  * pos_y_ * pos_z_;
    scalar_t bse = pos_x_ * pos_y_ * pos_z_;

    scalar_t dL_dix = static_cast<scalar_t>(0), dL_diy = static_cast<scalar_t>(0), dL_diz = static_cast<scalar_t>(0);
    scalar_t *gOut_ptr_NCDHW = grad_output.data + n * gOut_sN + d * gOut_sD + h * gOut_sH + w * gOut_sW;
    printf( "\ngOut: %f\n", *gOut_ptr_NCDHW);
    index_t NC_offset_inp;
    if (input_requires_grad) {
      NC_offset_inp = n * gInp_sN;
    }
    scalar_t *inp_ptr_NC = input.data + n * inp_sN;
    // calculate bilinear weighted pixel value and set output pixel
    for (index_t c = 0; c < C; ++c, gOut_ptr_NCDHW += gOut_sC, NC_offset_inp += gInp_sC, inp_ptr_NC += inp_sC) {
      scalar_t gOut = *gOut_ptr_NCDHW;
      printf( "\n---gOut: %f\n", gOut);

      // calculate and set grad_input. See Note [Passing pointer and offset to at::native::fastAtomicAdd].
      if (input_requires_grad) {
        at::native::safe_add_3d(grad_input.data, _iz, _iy, _ix, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, tnw * gOut,
                    NC_offset_inp, grad_input_memory_span);
        at::native::safe_add_3d(grad_input.data, _iz, _iy, ix_, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, tne * gOut,
                    NC_offset_inp, grad_input_memory_span);
        at::native::safe_add_3d(grad_input.data, _iz, iy_, _ix, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, tsw * gOut,
                    NC_offset_inp, grad_input_memory_span);
        at::native::safe_add_3d(grad_input.data, _iz, iy_, ix_, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, tse * gOut,
                    NC_offset_inp, grad_input_memory_span);
        at::native::safe_add_3d(grad_input.data, iz_, _iy, _ix, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, bnw * gOut,
                    NC_offset_inp, grad_input_memory_span);
        at::native::safe_add_3d(grad_input.data, iz_, _iy, ix_, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, bne * gOut,
                    NC_offset_inp, grad_input_memory_span);
        at::native::safe_add_3d(grad_input.data, iz_, iy_, _ix, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, bsw * gOut,
                    NC_offset_inp, grad_input_memory_span);
        at::native::safe_add_3d(grad_input.data, iz_, iy_, ix_, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, bse * gOut,
                    NC_offset_inp, grad_input_memory_span);
      }
      // calculate grad_grid
      if (at::native::within_bounds_3d(_iz, _iy, _ix, inp_D, inp_H, inp_W)) {
        scalar_t tnw_val = inp_ptr_NC[_iz * inp_sD + _iy * inp_sH + _ix * inp_sW];
        dL_dix -= tnw_val * (pos_y) * (pos_z) * gOut;
        dL_diy -= tnw_val * (pos_x) * (pos_z) * gOut;
        dL_diz -= tnw_val * (pos_x) * (pos_y) * gOut;
      }
      if (at::native::within_bounds_3d(_iz, _iy, ix_, inp_D, inp_H, inp_W)) {
        scalar_t tne_val = inp_ptr_NC[_iz * inp_sD + _iy * inp_sH + ix_ * inp_sW];
        dL_dix += tne_val * (pos_y) * (pos_z) * gOut;
        dL_diy -= tne_val * (pos_x_) * (pos_z) * gOut;
        dL_diz -= tne_val * (pos_x_) * (pos_y) * gOut;
      }
      if (at::native::within_bounds_3d(_iz, iy_, _ix, inp_D, inp_H, inp_W)) {
        scalar_t tsw_val = inp_ptr_NC[_iz * inp_sD + iy_ * inp_sH + _ix * inp_sW];
        dL_dix -= tsw_val * (pos_y_) * (pos_z) * gOut;
        dL_diy += tsw_val * (pos_x) * (pos_z) * gOut;
        dL_diz -= tsw_val * (pos_x) * (pos_y_) * gOut;
      }
      if (at::native::within_bounds_3d(_iz, iy_, ix_, inp_D, inp_H, inp_W)) {
        scalar_t tse_val = inp_ptr_NC[_iz * inp_sD + iy_ * inp_sH + ix_ * inp_sW];
        dL_dix += tse_val * (pos_y_) * (pos_z) * gOut;
        dL_diy += tse_val * (pos_x_) * (pos_z) * gOut;
        dL_diz -= tse_val * (pos_x_) * (pos_y_) * gOut;
      }
      if (at::native::within_bounds_3d(iz_, _iy, _ix, inp_D, inp_H, inp_W)) {
        scalar_t bnw_val = inp_ptr_NC[iz_ * inp_sD + _iy * inp_sH + _ix * inp_sW];
        dL_dix -= bnw_val * (pos_y) * (pos_z_) * gOut;
        dL_diy -= bnw_val * (pos_x) * (pos_z_) * gOut;
        dL_diz += bnw_val * (pos_x) * (pos_y) * gOut;
      }
      if (at::native::within_bounds_3d(iz_, _iy, ix_, inp_D, inp_H, inp_W)) {
        scalar_t bne_val = inp_ptr_NC[iz_ * inp_sD + _iy * inp_sH + ix_ * inp_sW];
        dL_dix += bne_val * (pos_y) * (pos_z_) * gOut;
        dL_diy -= bne_val * (pos_x_) * (pos_z_) * gOut;
        dL_diz += bne_val * (pos_x_) * (pos_y) * gOut;
      }
      if (at::native::within_bounds_3d(iz_, iy_, _ix, inp_D, inp_H, inp_W)) {
        scalar_t bsw_val = inp_ptr_NC[iz_ * inp_sD + iy_ * inp_sH + _ix * inp_sW];
        dL_dix -= bsw_val * (pos_y_) * (pos_z_) * gOut;
        dL_diy += bsw_val * (pos_x) * (pos_z_) * gOut;
        dL_diz += bsw_val * (pos_x) * (pos_y_) * gOut;
      }
      if (at::native::within_bounds_3d(iz_, iy_, ix_, inp_D, inp_H, inp_W)) {
        scalar_t bse_val = inp_ptr_NC[iz_ * inp_sD + iy_ * inp_sH + ix_ * inp_sW];
        dL_dix += bse_val * (pos_y_) * (pos_z_) * gOut;
        dL_diy += bse_val * (pos_x_) * (pos_z_) * gOut;
        dL_diz += bse_val * (pos_x_) * (pos_y_) * gOut;
      }
    }

    // assuming grad_grid is contiguous
    // thus we can
    //   1. use index with gGrid_sW to directly compute gGrid_ptr_NDHW
    //   2. directly assign to gGrid_ptr_NDHW[0], gGrid_ptr_NDHW[1], gGrid_ptr_NDHW[2]
    printf("dL_dix_mult=%f  dL_dix_mult=%f  dL_dix_mult=%f \n",dL_dix_mult,dL_diy_mult,dL_diz_mult);
    printf("\ndL_dix=%f",dL_dix);
    scalar_t *gGrid_ptr_NDHW = grad_grid.data + index * gGrid_sW;
    gGrid_ptr_NDHW[0] = dL_dix_mult * dL_dix * pos_x_derivative;
    gGrid_ptr_NDHW[1] = dL_diy_mult * dL_diy * pos_y_derivative;
    gGrid_ptr_NDHW[2] = dL_diz_mult * dL_diz * pos_z_derivative;
    printf("\ngGrid_ptr_NDHW=%f,%f,%f\n",gGrid_ptr_NDHW[0],gGrid_ptr_NDHW[1],gGrid_ptr_NDHW[2]);
  }
}


template <typename scalar_t, typename index_t>
C10_LAUNCH_BOUNDS_1(256)
__global__ void smooth_sampler_backward_backward_kernel(
    const index_t nthreads,
    at::cuda::detail::TensorInfo<scalar_t, index_t> grad_input, // initialized to empty
    at::cuda::detail::TensorInfo<scalar_t, index_t> grad_grid, // initialized to zeros
    at::cuda::detail::TensorInfo<scalar_t, index_t> grad_grad_out, // initialized to zeros
    at::cuda::detail::TensorInfo<scalar_t, index_t> input,
    at::cuda::detail::TensorInfo<scalar_t, index_t> grid,
    at::cuda::detail::TensorInfo<scalar_t, index_t> grad_out_input,
    at::cuda::detail::TensorInfo<scalar_t, index_t> grad_out_grid,
    at::cuda::detail::TensorInfo<scalar_t, index_t> grad_output,
    const at::native::detail::GridSamplerPadding padding_mode,
    bool align_corners,
    bool apply_smoothstep,
    bool input_requires_grad,
    const index_t grad_input_memory_span,
    const index_t grad_grad_out_memory_span) {
  // printf("----------2-----------\n");
  // printf( "%f, %f, %f, %f, %f\n", input.sizes[0],input.sizes[1],input.sizes[2],input.sizes[3],input.sizes[4] );
  // printf( "%f, %f, %f, %f, %f\n", grid.sizes[0],grid.sizes[1],grid.sizes[2],grid.sizes[3],grid.sizes[4] );
  index_t C = input.sizes[1];
  index_t inp_D = input.sizes[2];
  index_t inp_H = input.sizes[3];
  index_t inp_W = input.sizes[4];
  index_t out_D = grid.sizes[1];
  index_t out_H = grid.sizes[2];
  index_t out_W = grid.sizes[3];
  index_t inp_sN = input.strides[0]; //3x4x5
  index_t inp_sC = input.strides[1]; //3x4x5
  index_t inp_sD = input.strides[2]; //4x5
  index_t inp_sH = input.strides[3]; //5
  index_t inp_sW = input.strides[4]; //1
  index_t grid_sN = grid.strides[0]; //10x3
  index_t grid_sD = grid.strides[1]; //3
  index_t grid_sH = grid.strides[2]; //3
  index_t grid_sW = grid.strides[3]; //3 
  index_t grid_sCoor = grid.strides[4]; //1

  index_t gGrid_sW = grad_grid.strides[3];

  index_t gOut_sN = grad_output.strides[0];
  index_t gOut_sC = grad_output.strides[1];
  index_t gOut_sD = grad_output.strides[2];
  index_t gOut_sH = grad_output.strides[3];
  index_t gOut_sW = grad_output.strides[4];

  index_t gOutGrid_sW = grad_out_grid.strides[3];

  index_t gOutInput_sN = 0;
  index_t gOutInput_sC = 0;

  if (input_requires_grad) {
    gOutInput_sN = grad_out_input.strides[0];
    gOutInput_sC = grad_out_input.strides[1];
  }

  index_t gInp_sN = grad_input.strides[0];
  index_t gInp_sC = grad_input.strides[1];
  index_t gInp_sD = grad_input.strides[2];
  index_t gInp_sH = grad_input.strides[3];
  index_t gInp_sW = grad_input.strides[4];

  CUDA_KERNEL_LOOP_TYPE(index, nthreads, index_t) {
    const index_t w = index % out_W;
    const index_t h = (index / out_W) % out_H;
    const index_t d = (index / (out_H * out_W)) % out_D;
    const index_t n = index / (out_D * out_H * out_W);
    const auto grid_offset = n * grid_sN + d * grid_sD + h * grid_sH + w * grid_sW;
    // printf( "%f, %f, %f, %f\n", w,h,d,n );
    // printf( "%f, %f, %f, %f\n", grid_sN,grid_sD,grid_sH,grid_sW );

    // get the corresponding input x, y, z co-ordinates from grid
    scalar_t ix = grid.data[grid_offset];
    scalar_t iy = grid.data[grid_offset + grid_sCoor];
    scalar_t iz = grid.data[grid_offset + 2 * grid_sCoor];

    // multipliers for gradients on ix, iy, and iz
    scalar_t dL_dix_mult, dL_diy_mult, dL_diz_mult;
    ix = at::native::grid_sampler_compute_source_index_set_grad(ix, inp_W, padding_mode, align_corners, &dL_dix_mult); 
    iy = at::native::grid_sampler_compute_source_index_set_grad(iy, inp_H, padding_mode, align_corners, &dL_diy_mult);
    iz = at::native::grid_sampler_compute_source_index_set_grad(iz, inp_D, padding_mode, align_corners, &dL_diz_mult);

    // get corner pixel values from (x, y, z)
    // for 4d, we used north-east-south-west
    // for 5d, we add top-bottom
    index_t _ix = static_cast<index_t>(::floor(ix));
    index_t _iy = static_cast<index_t>(::floor(iy));
    index_t _iz = static_cast<index_t>(::floor(iz));
    index_t ix_ = _ix + 1;
    index_t iy_ = _iy + 1;
    index_t iz_ = _iz + 1;

    scalar_t pos_x_ = ix - _ix;
    scalar_t pos_y_ = iy - _iy;
    scalar_t pos_z_ = iz - _iz;

    scalar_t pos_x_derivative_ = dL_dix_mult;
    scalar_t pos_y_derivative_ = dL_diy_mult;
    scalar_t pos_z_derivative_ = dL_diz_mult;

    scalar_t pos_x_2nd_derivative_ = 0.0f;
    scalar_t pos_y_2nd_derivative_ = 0.0f;
    scalar_t pos_z_2nd_derivative_ = 0.0f;

    scalar_t pos_x_2nd_derivative = 0.0f;
    scalar_t pos_y_2nd_derivative = 0.0f;
    scalar_t pos_z_2nd_derivative = 0.0f;

    if (apply_smoothstep) {
      pos_x_derivative_ *= smoothstep_derivative(pos_x_);
      pos_y_derivative_ *= smoothstep_derivative(pos_y_);
      pos_z_derivative_ *= smoothstep_derivative(pos_z_);

      pos_x_2nd_derivative_ = dL_dix_mult * dL_dix_mult * smoothstep_2nd_derivative(pos_x_);
      pos_y_2nd_derivative_ = dL_diy_mult * dL_diy_mult * smoothstep_2nd_derivative(pos_y_);
      pos_z_2nd_derivative_ = dL_diz_mult * dL_diz_mult * smoothstep_2nd_derivative(pos_z_);

      pos_x_2nd_derivative = -pos_x_2nd_derivative_;
      pos_y_2nd_derivative = -pos_y_2nd_derivative_;
      pos_z_2nd_derivative = -pos_z_2nd_derivative_;

      pos_x_ = smoothstep(pos_x_);
      pos_y_ = smoothstep(pos_y_);
      pos_z_ = smoothstep(pos_z_);
    }

    scalar_t pos_x = 1.0f - pos_x_;
    scalar_t pos_y = 1.0f - pos_y_;
    scalar_t pos_z = 1.0f - pos_z_;

    scalar_t pos_x_derivative = -pos_x_derivative_;
    scalar_t pos_y_derivative = -pos_y_derivative_;
    scalar_t pos_z_derivative = -pos_z_derivative_;

    index_t index_corners[2][3] = {{_ix, _iy, _iz},
                                   {ix_, iy_, iz_}};
    scalar_t pos_corners[2][9] = {{pos_x, pos_y, pos_z,
                                   pos_x_derivative, pos_y_derivative, pos_z_derivative,
                                   pos_x_2nd_derivative, pos_y_2nd_derivative, pos_z_2nd_derivative},
                                  {pos_x_, pos_y_, pos_z_,
                                   pos_x_derivative_, pos_y_derivative_, pos_z_derivative_,
                                   pos_x_2nd_derivative_, pos_y_2nd_derivative_, pos_z_2nd_derivative_}};
    scalar_t surface_coefficients[8] = {};
    scalar_t out_derivatives[8][12] = {};

    #pragma unroll
    for (int shift = 0; shift < 8; shift++) {
      int px = (shift >> 0) & 1;
      int py = (shift >> 1) & 1;
      int pz = (shift >> 2) & 1;

      surface_coefficients[shift] = pos_corners[px][0] * pos_corners[py][1] * pos_corners[pz][2];

      out_derivatives[shift][0] = pos_corners[py][1] * pos_corners[pz][2] * pos_corners[px][3]; // dOut_dx / surf_weight
      out_derivatives[shift][1] = pos_corners[py][1] * pos_corners[pz][2] * pos_corners[px][6]; // d2Out_dx2 / surf_weight
      out_derivatives[shift][2] = pos_corners[py][4] * pos_corners[pz][2] * pos_corners[px][3]; // d2Out_dxdy / surf_weight
      out_derivatives[shift][3] = pos_corners[py][1] * pos_corners[pz][5] * pos_corners[px][3]; // d2Out_dxdz / surf_weight

      out_derivatives[shift][4] = pos_corners[px][0] * pos_corners[pz][2] * pos_corners[py][4]; // dOut_dy / surf_weight
      out_derivatives[shift][5] = pos_corners[px][0] * pos_corners[pz][2] * pos_corners[py][7]; // d2Out_dy2 / surf_weight
      out_derivatives[shift][6] = pos_corners[px][3] * pos_corners[pz][2] * pos_corners[py][4]; // d2Out_dydx / surf_weight
      out_derivatives[shift][7] = pos_corners[px][0] * pos_corners[pz][5] * pos_corners[py][4]; // d2Out_dydz / surf_weight

      out_derivatives[shift][8] = pos_corners[px][0] * pos_corners[py][1] * pos_corners[pz][5]; // dOut_dz / surf_weight
      out_derivatives[shift][9] = pos_corners[px][0] * pos_corners[py][1] * pos_corners[pz][8]; // d2Out_dz2 / surf_weight
      out_derivatives[shift][10] = pos_corners[px][3] * pos_corners[py][1] * pos_corners[pz][5]; // d2Out_dzdx / surf_weight
      out_derivatives[shift][11] = pos_corners[px][0] * pos_corners[py][4] * pos_corners[pz][5]; // d2Out_dzdy / surf_weight
    }

    scalar_t d2L_dix2 = static_cast<scalar_t>(0), d2L_diy2 = static_cast<scalar_t>(0), d2L_diz2 = static_cast<scalar_t>(0);
    index_t offset_out_DHW = d * gOut_sD + h * gOut_sH + w * gOut_sW;
    scalar_t *gOut_ptr_NCDHW = grad_output.data + n * gOut_sN + offset_out_DHW;
    index_t NC_offset_inp = n * gInp_sN;
    index_t NC_offset_out = n * gOut_sN;
    scalar_t *inp_ptr_NC = input.data + n * inp_sN;

    scalar_t *gOutInput_ptr_NC = NULL;

    if (input_requires_grad) {
      gOutInput_ptr_NC = grad_out_input.data + n * gOutInput_sN;
    }

    scalar_t *gOutGrid_ptr_NDHW = grad_out_grid.data + index * gOutGrid_sW;
    scalar_t *gGrid_ptr_NDHW = grad_grid.data + index * gGrid_sW;

    for (index_t c = 0; c < C; ++c, gOut_ptr_NCDHW += gOut_sC, inp_ptr_NC += inp_sC, gOutInput_ptr_NC += gOutInput_sC, NC_offset_inp += gInp_sC, NC_offset_out += gOut_sC) {
      scalar_t gOut = *gOut_ptr_NCDHW;

      #pragma unroll
      for (int shift = 0; shift < 8; shift++) {
        int px = (shift >> 0) & 1;
        int py = (shift >> 1) & 1;
        int pz = (shift >> 2) & 1;

        index_t ix = index_corners[px][0];
        index_t iy = index_corners[py][1];
        index_t iz = index_corners[pz][2];

        // Slightly unprecise naming: in fact these are divided by surf_weight.
        scalar_t dOut_dx = out_derivatives[shift][0]; // E.g. variable "dOut_dx" is mathematically "dOut/dx * 1/surf_weight"
        scalar_t d2Out_dx2 = out_derivatives[shift][1];
        scalar_t d2Out_dxdy = out_derivatives[shift][2];
        scalar_t d2Out_dxdz = out_derivatives[shift][3];
        scalar_t dOut_dy = out_derivatives[shift][4];
        scalar_t d2Out_dy2 = out_derivatives[shift][5];
        scalar_t d2Out_dydx = out_derivatives[shift][6];
        scalar_t d2Out_dydz = out_derivatives[shift][7];
        scalar_t dOut_dz = out_derivatives[shift][8];
        scalar_t d2Out_dz2 = out_derivatives[shift][9];
        scalar_t d2Out_dzdx = out_derivatives[shift][10];
        scalar_t d2Out_dzdy = out_derivatives[shift][11];
        scalar_t surface_coeff = surface_coefficients[shift];

        if (at::native::within_bounds_3d(iz, iy, ix, inp_D, inp_H, inp_W)) {
          index_t inp_el = iz * inp_sD + iy * inp_sH + ix * inp_sW;
          scalar_t surf_weight = inp_ptr_NC[inp_el];

          scalar_t dL_dx = gOut * dOut_dx;
          scalar_t dL_dy = gOut * dOut_dy;
          scalar_t dL_dz = gOut * dOut_dz;

          scalar_t gOutGrid_x = gOutGrid_ptr_NDHW[0];
          scalar_t gOutGrid_y = gOutGrid_ptr_NDHW[1];
          scalar_t gOutGrid_z = gOutGrid_ptr_NDHW[2];

          scalar_t grad_grad_out_delta = surf_weight * (dOut_dx * gOutGrid_x
                                                       + dOut_dy * gOutGrid_y
                                                       + dOut_dz * gOutGrid_z);

          if (gOutInput_ptr_NC != NULL) {
            scalar_t gOutInput = gOutInput_ptr_NC[inp_el];
            grad_grad_out_delta += gOutInput * surface_coeff;
            d2L_dix2 += dL_dx * gOutInput;
            d2L_diy2 += dL_dy * gOutInput;
            d2L_diz2 += dL_dz * gOutInput;
          }

          at::native::fastAtomicAdd(grad_grad_out.data,
                                    NC_offset_out + offset_out_DHW,
                                    grad_grad_out_memory_span,
                                    grad_grad_out_delta,
                                    true);

          d2L_dix2 += surf_weight * gOut * (d2Out_dx2 * gOutGrid_x
                                            + d2Out_dxdy * gOutGrid_y
                                            + d2Out_dxdz * gOutGrid_z);
          d2L_diy2 += surf_weight * gOut * (d2Out_dydx * gOutGrid_x
                                            + d2Out_dy2 * gOutGrid_y
                                            + d2Out_dydz * gOutGrid_z);
          d2L_diz2 += surf_weight * gOut * (d2Out_dzdx * gOutGrid_x
                                            + d2Out_dzdy * gOutGrid_y
                                            + d2Out_dz2 * gOutGrid_z);
          
          add_3d(grad_input.data, iz, iy, ix, gInp_sD, gInp_sH, gInp_sW,
                dL_dx * gOutGrid_x + dL_dy * gOutGrid_y + dL_dz * gOutGrid_z,
                NC_offset_inp, grad_input_memory_span);
        }
      }
    }

    gGrid_ptr_NDHW[0] = d2L_dix2;
    gGrid_ptr_NDHW[1] = d2L_diy2;
    gGrid_ptr_NDHW[2] = d2L_diz2;
  }
}

void launch_smooth_sampler_forward_kernel(
    const torch::TensorBase &output, const torch::TensorBase &input, const torch::TensorBase &grid,
    int64_t padding_mode, bool align_corners, bool apply_smoothstep) {
  auto N = input.size(0); //[1,1,H,W,Z]
  auto D = grid.size(1); //[1,N,1,1,3]
  auto H = grid.size(2);
  auto W = grid.size(3);
  int64_t count = N * D * H * W;
  // printf( "%d, %d, %d, %d, %d", N, D, H, W, count);
  if (count > 0) {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "smooth_sampler_cuda", [&] {
      if (at::native::canUse32BitIndexMath(input) && at::native::canUse32BitIndexMath(grid) &&
          at::native::canUse32BitIndexMath(output)) {
        smooth_sampler_kernel<scalar_t>
          <<<at::cuda::detail::GET_BLOCKS(count, 512), 512, 0, at::cuda::getCurrentCUDAStream()>>>(
            static_cast<int>(count),
            at::cuda::detail::getTensorInfo<scalar_t, int>(input),
            at::cuda::detail::getTensorInfo<scalar_t, int>(grid),
            at::cuda::detail::getTensorInfo<scalar_t, int>(output),
            static_cast<at::native::detail::GridSamplerPadding>(padding_mode),
            align_corners,
            apply_smoothstep);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      } else {
        smooth_sampler_kernel<scalar_t>
          <<<at::cuda::detail::GET_BLOCKS(count, 512), 512, 0, at::cuda::getCurrentCUDAStream()>>>(
            count,
            at::cuda::detail::getTensorInfo<scalar_t, int64_t>(input),
            at::cuda::detail::getTensorInfo<scalar_t, int64_t>(grid),
            at::cuda::detail::getTensorInfo<scalar_t, int64_t>(output),
            static_cast<at::native::detail::GridSamplerPadding>(padding_mode),
            align_corners,
            apply_smoothstep);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }
    });
  }
}

// void launch_smooth_sampler_backward_kernel(
//     const torch::TensorBase &grad_input, const torch::TensorBase &grad_grid,
//     const torch::TensorBase& grad_output, const torch::TensorBase& input,
//     const torch::TensorBase& grid, int64_t padding_mode,
//     bool align_corners, bool apply_smoothstep, bool input_requires_grad) {
//   auto N = input.size(0);
//   auto D = grid.size(1);
//   auto H = grid.size(2);
//   auto W = grid.size(3);
//   int64_t count = N * D * H * W;
//   if (count > 0) {
//     AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "smooth_sampler_backward_cuda", [&] {
//       if (at::native::canUse32BitIndexMath(input) && at::native::canUse32BitIndexMath(grid) &&
//           at::native::canUse32BitIndexMath(grad_output)) {
//         smooth_sampler_backward_kernel<scalar_t>
//           <<<at::cuda::detail::GET_BLOCKS(count, 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
//             static_cast<int>(count),
//             at::cuda::detail::getTensorInfo<scalar_t, int>(grad_output),
//             at::cuda::detail::getTensorInfo<scalar_t, int>(input),
//             at::cuda::detail::getTensorInfo<scalar_t, int>(grid),
//             input_requires_grad ? at::cuda::detail::getTensorInfo<scalar_t, int>(grad_input) : at::cuda::detail::TensorInfo<scalar_t, int>(),
//             at::cuda::detail::getTensorInfo<scalar_t, int>(grad_grid),
//             static_cast<at::native::detail::GridSamplerPadding>(padding_mode),
//             align_corners,
//             apply_smoothstep,
//             /*grad_input_memory_span =*/input_requires_grad ? static_cast<int>(grad_input.numel()) : 0,
//             input_requires_grad);
//         C10_CUDA_KERNEL_LAUNCH_CHECK();
//       } else {
//         smooth_sampler_backward_kernel<scalar_t>
//           <<<at::cuda::detail::GET_BLOCKS(count, 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
//             count,
//             at::cuda::detail::getTensorInfo<scalar_t, int64_t>(grad_output),
//             at::cuda::detail::getTensorInfo<scalar_t, int64_t>(input),
//             at::cuda::detail::getTensorInfo<scalar_t, int64_t>(grid),
//             input_requires_grad ? at::cuda::detail::getTensorInfo<scalar_t, int64_t>(grad_input) : at::cuda::detail::TensorInfo<scalar_t, int64_t>(),
//             at::cuda::detail::getTensorInfo<scalar_t, int64_t>(grad_grid),
//             static_cast<at::native::detail::GridSamplerPadding>(padding_mode),
//             align_corners,
//             apply_smoothstep,
//             /*grad_input_memory_span =*/input_requires_grad ? grad_input.numel() : 0,
//             input_requires_grad);
//         C10_CUDA_KERNEL_LAUNCH_CHECK();
//       }
//     });
//   }
// }

//yx
void launch_smooth_sampler_backward_kernel(
  const torch::TensorBase &grad_grid,
  const torch::TensorBase& grad_output, const torch::TensorBase& delta, const torch::TensorBase& corner_val_8 //const torch::TensorBase& points, const torch::TensorBase& corners
) {
// auto N = points.size(0);
auto D = delta.size(1);
// auto H = points.size(2);
// auto W = points.size(3);
int64_t count = D; //*H* W;
// printf( "\n ---count=%d, ",count);
if (count > 0) {
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(delta.scalar_type(), "smooth_sampler_backward_cuda", [&] {
      // printf("  00--");
      smooth_sampler_backward_kernel2<scalar_t>
        <<<at::cuda::detail::GET_BLOCKS(count, 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
          static_cast<int>(count),
          at::cuda::detail::getTensorInfo<scalar_t, int>(grad_output),
          // at::cuda::detail::getTensorInfo<scalar_t, int>(corners),
          // at::cuda::detail::getTensorInfo<scalar_t, int>(points),
          at::cuda::detail::getTensorInfo<scalar_t, int>(delta),
          // at::cuda::detail::getTensorInfo<scalar_t, int>(grad_point),
          at::cuda::detail::getTensorInfo<scalar_t, int>(corner_val_8),
          // input_requires_grad ? at::cuda::detail::getTensorInfo<scalar_t, int>(grad_input) : at::cuda::detail::TensorInfo<scalar_t, int>(),
          at::cuda::detail::getTensorInfo<scalar_t, int>(grad_grid)
         );
        //  cudaDeviceSynchronize();
        // printf("CUDA error: %s\n", cudaGetErrorString(cudaGetLastError()));
        // printf("11--");
      C10_CUDA_KERNEL_LAUNCH_CHECK();
  });
}
}

//yx
void launch_smooth_sampler_backward_backward_kernel(
  const torch::TensorBase& grad_input,
  const torch::TensorBase& grad_grid,
  const torch::TensorBase& grad_grad_out,
  // const torch::TensorBase& points, 
  // const torch::TensorBase& corners, 
  const torch::TensorBase& delta,
  const torch::TensorBase& corner_val_8,
  const torch::TensorBase& grad_out_grid,
  const torch::TensorBase& grad_output) {
// auto N = input.size(0);
auto D = delta.size(1);
int64_t count = D;
if (count > 0) {
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(delta.scalar_type(), "smooth_sampler_backward_backward_cuda", [&] {
    // smooth_sampler_corner_kernel<scalar_t>
    // <<<at::cuda::detail::GET_BLOCKS(count, 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
    //   static_cast<int>(count),
    //   at::cuda::detail::getTensorInfo<scalar_t, int>(grad_output),
    //   at::cuda::detail::getTensorInfo<scalar_t, int>(input),
    //   at::cuda::detail::getTensorInfo<scalar_t, int>(grid),
    //   at::cuda::detail::getTensorInfo<scalar_t, int>(points),
    //   at::cuda::detail::getTensorInfo<scalar_t, int>(corners),
    //   at::cuda::detail::getTensorInfo<scalar_t, int>(grad_point),
    //   at::cuda::detail::getTensorInfo<scalar_t, int>(corner_val_8),
    //   at::cuda::detail::getTensorInfo<scalar_t, int>(grad_input),
    //   static_cast<at::native::detail::GridSamplerPadding>(padding_mode),
    //   align_corners,
    //   static_cast<int>(grad_input.numel()));
    // printf("22--");
    smooth_sampler_backward_backward_kernel2<scalar_t>
    <<<at::cuda::detail::GET_BLOCKS(count, 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      static_cast<int>(count),
      at::cuda::detail::getTensorInfo<scalar_t, int>(grad_input),
      at::cuda::detail::getTensorInfo<scalar_t, int>(grad_grid),
      at::cuda::detail::getTensorInfo<scalar_t, int>(grad_grad_out),
      // at::cuda::detail::getTensorInfo<scalar_t, int>(corners),
      // at::cuda::detail::getTensorInfo<scalar_t, int>(points),
      at::cuda::detail::getTensorInfo<scalar_t, int>(delta),
      // at::cuda::detail::getTensorInfo<scalar_t, int>(grad_point),
      at::cuda::detail::getTensorInfo<scalar_t, int>(corner_val_8), 
      // input_requires_grad ? at::cuda::detail::getTensorInfo<scalar_t, int>(grad_out_input) : at::cuda::detail::TensorInfo<scalar_t, int>(),
      at::cuda::detail::getTensorInfo<scalar_t, int>(grad_out_grid),
      at::cuda::detail::getTensorInfo<scalar_t, int>(grad_output),
      static_cast<int>(grad_input.numel()),
      static_cast<int>(grad_grad_out.numel()));
  // printf("33--");
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  });
}
}


}