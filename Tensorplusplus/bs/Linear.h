#pragma once

#include "../Tensor/Tensor.h"

namespace Tensorplusplus {
	namespace bs {
		template<typename T>
		class Linear{
		public:
			Tensor<T> weight;
			Tensor<T> bias;

			explicit Linear() = delete;

			explicit Linear(long long size_x, long long size_y, bool bias = true, Device device = Device::CUDA) {
				weight = Tensor<T>({ size_x,size_y }, device, true, true);
				weight.required_grad() = true;
				if (bias) {
					this->bias = Tensor<T>({ size_y }, device, true, true);
					this->bias.required_grad() = true;
				}
			}

			Tensor<T> forward(Tensor<T> tensor_in) {
				if (bias.initialize()) {
					return matmul(tensor_in, weight) + bias;
				}
				return matmul(tensor_in, weight);
			}
		};


	}
}