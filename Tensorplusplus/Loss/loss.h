#pragma once

#include "../Tensor/Tensor.h"

namespace Tensorplusplus {
	namespace loss {
		template<typename T>
		class loss {
		public:
			Tensor<T> loss_tensor;
			virtual void AGD() {
#ifdef _DEBUG
				if (loss_tensor.initialize() == false) {
					std::cout << "Loss_AGD_error: loss_tensor not initialize" << "\n";
					throw std::invalid_argument("Loss_AGD_error: loss_tensor not initialize");
				}
						
#endif
				loss_tensor.backward();
			}

		};

	}
}