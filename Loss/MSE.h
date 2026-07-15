#pragma once

#include "loss.h"

namespace Tensorplusplus {
	namespace loss {
		template<typename T>
		class MSE : public loss<T> {
		public:
			explicit MSE() = default;

			Tensor<T> forward(Tensor<T> X, Tensor<T> Y) {
#ifdef _DEBUG
				if (X.shape() != Y.shape()) {
					std::cout << "MSE_error-> X_shape:" << X.shape() << " Y_shape:" << Y.shape()<< "must equal" << "\n";
					throw std::invalid_argument("MSE_loss:X.shape must equal to Y.shape");
				}
#endif
				
				this->loss_tensor = pow((X - Y), 2);
				return this->loss_tensor;
			}

			T mean_forward(Tensor<T> X, Tensor<T> Y) {
				auto forward_tensor = forward(X, Y);
				this->loss_tensor = forward_tensor / T(forward_tensor.total_size());

				T result = cu_total_sum(this->loss_tensor) / T(this->loss_tensor.total_size());
				return result;
			}

			T sum_forward(Tensor<T> X, Tensor<T> Y) {
				this->loss_tensor = forward(X, Y);

				T result = cu_total_sum(this->loss_tensor);
				return result;
			}

		};
	}
}