#pragma once

#include "../Tensor/Tensor.h"

namespace Tensorplusplus {
	namespace function {
		template<typename T>
		class tanh {
		public:
			static Tensor<T> forward(Tensor<T> A) {
				return (exp(A) - exp(-A)) / (exp(A) + exp(-A));
			}

		};
	}
}