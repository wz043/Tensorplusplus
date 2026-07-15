#pragma once

#include "../Tensor/Tensor.h"

namespace Tensorplusplus {
	namespace bs {
		template<typename T>
		class RMSNorm {
		private:
			long long norm_dim;
			T eps;
			Tensor<T> gamma;
		public:
			explicit RMSNorm() = delete;

			explicit RMSNorm(long long norm_dim, std::vector<long long> gamma_shape, T eps = T(1e-6), Device device = Device::CUDA) :norm_dim(norm_dim), eps(eps) {
				this->gamma = Tensor<T>(gamma_shape, device, false, true);
				gamma.One();
				gamma.required_grad() = true;
			}

			Tensor<T> forward(Tensor<T> A) {
				auto norm_dim_in_shape = A.shape()[this->norm_dim];
				auto RMS = sqrt((pow(A, 2)).sum_in_dim(this->norm_dim) / T(norm_dim_in_shape) + this->eps);
				return gamma * (A / RMS.broadcast_in_dim(this->norm_dim, norm_dim_in_shape));
			}
		};
	}
}