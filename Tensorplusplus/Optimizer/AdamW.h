#pragma once

#include "optimizer.h"

namespace Tensorplusplus {
	namespace optimizer {
		template<typename T>
		class AdamW : public optimizer<T> {
		private:
			float b1;
			float b2;
			float eps;
			float wd;
			bool init = false;

			float b1t = 1.0f;
			float b2t = 1.0f;
		public:
			std::vector<Tensor<T>> total_v;
			std::vector<Tensor<T>> total_m;

			explicit AdamW() = delete;

			explicit AdamW(float beta1, float beta2, float eps, float weight_decay) {
				b1 = beta1;
				b2 = beta2;
				this->eps = eps;
				wd = weight_decay;
			}

			void Update(T lr) override {
				if (init == false) {
					total_v = std::vector<Tensor<T>>(bs::module<T>::parameter_lists().size());
					for (int i = 0; i < total_v.size(); i++) {
						total_v[i] = Tensor<T>(bs::module<T>::parameter_lists()[i].shape(), bs::module<T>::parameter_lists()[i].device(), false);
					}

					total_m = std::vector<Tensor<T>>(bs::module<T>::parameter_lists().size());
					for (int i = 0; i < total_m.size(); i++) {
						total_m[i] = Tensor<T>(bs::module<T>::parameter_lists()[i].shape(), bs::module<T>::parameter_lists()[i].device(), false);
					}
					init = true;
				}

				for (int i = 0; i < bs::module<T>::parameter_lists().size(); i++) {
					auto& P = bs::module<T>::parameter_lists()[i];
					auto& m = total_m[i];
					auto& v = total_v[i];
					

					m = T(b1) * m + T(1.0f - b1) * P.grad();
					v = T(b2) * v + T(1.0f - b2) * pow(P.grad(), 2);

					this->b1t *= b1;
					this->b2t *= b2;

					m = m / T(1.0f - b1t);
					v = v / T(1.0f - b2t);

					P -= lr * (m / (sqrt(v) + eps) + wd * P);

				}
				
			}

		};

	}
}