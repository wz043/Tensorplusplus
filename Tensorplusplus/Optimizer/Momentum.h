#pragma once

#include "optimizer.h"

namespace Tensorplusplus {
	namespace optimizer {
		template<typename T>
		class Momentum : public optimizer<T> {
		private:
			bool init = false;

		public:
			std::vector<Tensor<T>> total_v;
			T b;

			explicit Momentum() = delete;

			explicit Momentum(T b) {
				this->b = b;
			}

			void Update(T lr) {
				if (init == false) {
					total_v = std::vector<Tensor<T>>(bs::module<T>::parameter_lists().size());
					for (int i = 0; i < total_v.size(); i++) {
						total_v[i] = Tensor<T>(bs::module<T>::parameter_lists()[i].shape(), bs::module<T>::parameter_lists()[i].device(),false);
					}
					init = true;
				}

				for (int i = 0; i < total_v.size(); i++) {
					auto& P = bs::module<T>::parameter_lists()[i];
					auto& v = total_v[i];
					v = b * v - lr * P.grad();
					P += v;
				
				}
			}

		};
	}

}