#pragma once

#include "../Tensor/Tensor.h"

namespace Tensorplusplus {
	namespace function {
		template<typename T>
		class sigmoid {
		public:
			static Tensor<T> forward(Tensor<T> A) {
				return T(1) / (T(1) + exp(-A));
			}
		};
	}
}