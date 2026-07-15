#pragma once

#include "../Tensor/Tensor.h"

namespace Tensorplusplus {
	namespace bs {
		template<typename T>
		class Conv2D {
		public:
			Tensor<T> kernel;
			Tensor<T> bias;
			std::vector<long long> stride;
			std::vector<long long> padding;

			explicit Conv2D() = delete;

			explicit Conv2D(long long in_channel, long long out_channel, std::vector<long long> kernel_size, std::vector<long long> stride, std::vector<long long> padding = {0,0}, bool bias = true, Device device = Device::CUDA) {
				kernel = Tensor<T>({ out_channel,in_channel,kernel_size[0],kernel_size[1] }, device, true, true);
				kernel.required_grad() = true;
				this->stride = stride;
				this->padding = padding;
				if (bias) {
					this->bias = Tensor<T>({ out_channel }, device, true, true);
					this->bias.required_grad() = true;
				}
			}


			Tensor<T> forward(Tensor<T> A) {
				if (bias.initialize()) {
					Tensor<T> output = conv2d(A, kernel, this->stride, this->padding);
					Tensor<T> bias = this->bias.broadcast_in_dim(1, output.shape()[2]);
					return output + bias.broadcast_in_dim(2, output.shape()[3]);
				}
				return conv2d(A, kernel, this->stride, this->padding);

			}
		};
	}
}