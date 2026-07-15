#pragma once

#include "../Tensor/Tensor.h"

namespace Tensorplusplus {
	namespace function {
		template<typename T>
		class softmax {
		public:
			static Tensor<T> forward(Tensor<T> A, long long dim) {
				return exp(A) / (exp(A).sum_in_dim(dim)).broadcast_in_dim(dim, A.shape()[dim]);
			}

		};
	}
}