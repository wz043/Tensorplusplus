#pragma once

#include "error_states.h"

//DEFINE

//cuda kernel block size
static const long long block_x = 16;
static const long long block_y = 16;
static const long long block1D = 256;

namespace cu_plusplus {
	extern "C" {
		__declspec(dllimport) int cu_memory_inquire(double& total_memory, double& free_memory);

		__declspec(dllimport) int cu_memory_apply(void*& ptr, long long bytes_total_size, DType dtype);

		__declspec(dllimport) int cu_memory_free(void*& ptr);
		//Zero
		__declspec(dllimport) int cu_Zero(void* A, long long total_size, DType dtype);

		//One
		__declspec(dllimport) int cu_One(void* A, long long total_size, DType dtype);

		__declspec(dllimport) int cu_Copy(void* A, void* B, long long total_size, DType dtype);

		//rand_init
		__declspec(dllimport) int cu_rand_init(void* A, long long total_size, float E, float S, long long seed, DType dtype);

		//batch_down
		__declspec(dllimport) int cu_batch_down(void* A, void* B, long long A_total_size, long long B_total_size, DType dtype);

		//ADD
		__declspec(dllimport) int cu_add(void* A, void* B, void* C, long long A_total_size, long long B_total_size, DType dtype);

		//add_with_const
		__declspec(dllimport) int cu_add_with_const(void* A, void* a, void* B, long long total_size, DType dtype);

		//Opposite
		__declspec(dllimport) int cu_opposite(void* A, void* B, long long total_size, DType dtype);

		//Sub
		__declspec(dllimport) int cu_sub(void* A, void* B, void* C, long long A_total_size, long long B_total_size, DType dtype);

		//Sub_with_rconst
		__declspec(dllimport) int cu_sub_with_rconst(void* A, void* a, void* B, long long total_size, DType dtype);

		//sub_with_lconst
		__declspec(dllimport) int cu_sub_with_lconst(void* a, void* A, void* B, long long total_size, DType dtype);

		//Transpose
		__declspec(dllimport) int cu_transpose(void* A, void* B, long long m, long long n, long long A_batch, DType dtype);

		//transpose_with_dim
		__declspec(dllimport) int cu_transpose_with_dim(void* A, void* B, long long B_total_size, long long dim1_in_shape, long long between_batch, long long dim2_in_shape, long long behind_batch, DType dtype);


		//cublas_handle_create
		__declspec(dllimport) int cublas_handle_create(void*& handle);

		//cublas_handle_destroy
		__declspec(dllimport) int cublas_handle_destroy(void*& handle);

		//cublas_mamtul
		__declspec(dllimport) int cublas_matmul(void* cublas_handle, void* A, void* B, void* C, long long A_batch, long long B_batch, long long m, long long n, long long k, DType dtype);

		//Hadamard
		__declspec(dllimport) int cu_hadamard(void* A, void* B, void* C, long long A_total_size, long long B_total_size, DType dtype);

		//Hadamard_with_const
		__declspec(dllimport) int cu_hadamard_with_const(void* A, void* b, void* C, long long total_size, DType dtype);


		//Division
		__declspec(dllimport) int cu_division(void* A, void* B, void* C, long long A_total_size, long long B_total_size, DType dtype);

		//Division_with_rconst
		__declspec(dllimport) int cu_division_with_rconst(void* A, void* b, void* C, long long total_size, DType dtype);

		//Division_with_lconst
		__declspec(dllimport) int cu_division_with_lconst(void* a, void* B, void* C, long long total_size, DType dtype);

		//exp
		__declspec(dllimport) int cu_exp(void* A, void* B, long long total_size, DType dtype);

		//sqrt
		__declspec(dllimport) int cu_sqrt(void* A, void* B, long long total_size, DType dtype);

		//ln
		__declspec(dllimport) int cu_ln(void* A, void* B, long long total_size, DType dtype);

		//cos
		__declspec(dllimport) int cu_cos(void* A, void* B, long long total_size, DType dtype);

		//sin
		__declspec(dllimport) int cu_sin(void* A, void* B, long long total_size, DType dtype);

		//dim_up
		__declspec(dllimport) int cu_dim_up(void* A, void* B, long long B_total_size, long long back_total_size, long long up_dim_in_shape, DType dtype);

		//sum_in_dim
		__declspec(dllimport) int cu_sum_in_dim(void* A, void* B, long long B_total_size, long long back_total_size, long long dim_in_shape, DType dtype);

		//cu_total_sum
		__declspec(dllimport) int cu_total_sum(void* handle, void* A, void* sum, long long A_total_size, DType dtype);

		//ref_operation
		__declspec(dllimport) int cu_ref_add(void* A, void* B, long long A_total_size, long long B_total_size, DType dtype);

		__declspec(dllimport) int cu_ref_sub(void* A, void* B, long long A_total_size, long long B_total_size, DType dtype);

		__declspec(dllimport) int cu_ref_hadamard(void* A, void* B, long long A_total_size, long long B_total_size, DType dtype);

		__declspec(dllimport) int cu_ref_division(void* A, void* B, long long A_total_size, long long B_total_size, DType dtype);
		//cudnn_handle
		__declspec(dllimport) int cudnn_handle_create(void*& handle);

		//cudnn_handle_destroy
		__declspec(dllimport) int cudnn_handle_destroy(void*& handle);

		//convolution
		__declspec(dllimport) int cudnn_conv2d_forward(void* cudnn_handle, void* A, void* kernel, void* C, long long A_batch, long long in_channel, long long out_channel, long long A_x, long long A_y, long long kernel_x, long long kernel_y, long long stride_x, long long stride_y, long long padding_x, long long padding_y, DType dtype);

		__declspec(dllimport) int cudnn_conv2d_backward_data(void* cudnn_handle, void* backward_in_grad, void* kernel, void* backward_out_grad, long long batch, long long forward_in_channel, long long forward_out_channel, long long in_grad_x, long long in_grad_y, long long out_grad_x, long long out_grad_y, long long kernel_x, long long kernel_y, long long stride_x, long long stride_y, long long padding_x, long long padding_y, DType dtype);

		__declspec(dllimport) int cudnn_conv2d_backward_kernel(void* cudnn_handle, void* backward_in_grad, void* forward_in_tensor, void* kernel_grad, long long batch, long long forward_in_grad_channel, long long forward_out_grad_channel, long long in_grad_x, long long in_grad_y, long long in_tensor_x, long long in_tensor_y, long long kernel_x, long long kernel_y, long long stride_x, long long stride_y, long long padding_x, long long padding_y, DType dtype);

		__declspec(dllimport) int cu_ppo_clip(void* A, void* B, void* gd_mask, void* down_clip, void* up_clip, long long A_total_size, DType dtype);

		__declspec(dllimport) int cu_min(void* A, void* B, void* C, void* A_gdmask, void* B_gdmask, long long total_size, DType dtype);

		__declspec(dllimport) int cu_max(void* A, void* B, void* C, void* A_gdmask, void* B_gdmask, long long total_size, DType dtype);

		__declspec(dllimport) int cu_slice(void* A, void* B, long long B_total_size, long long A_origin_dim, long long B_slice_dim, long long dim_back_batch, long long slice_start, long long stride, DType dtype);

		__declspec(dllimport) int cu_slice_backward(void* A, void* B, long long A_total_size, long long B_origin_dim, long long A_slice_dim, long long dim_back_batch, long long slice_start, long long stride, DType dtype);
	}
}
