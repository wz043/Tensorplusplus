#pragma once

#include "../Tensor/Tensor.h"
#include "sigmoid.h"

namespace Tensorplusplus {
	namespace function {
		template<typename T>
		class silu {
		public:
			static Tensor<T> forward(Tensor<T> A) {
				return (A * sigmoid<T>::forward(A));
			}

		};
	}

}