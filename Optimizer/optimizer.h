#pragma once

#include "../Tensor/Tensor.h"
#include "../bs/module.h"

namespace Tensorplusplus {
	namespace optimizer {
		template<typename T>
		class optimizer {
		public:
			virtual void Update(T lr) = 0;

			void Grad_Zero() {
				for (int i = 0; i < bs::module<T>::parameter_lists().size(); i++) {
					(bs::module<T>::parameter_lists()[i]).grad().Zero();
				}
			}

			T L2_grad_norm() {
				T grad_norm2 = T(0);
				for (int i = 0; i < bs::module<T>::parameter_lists().size(); i++) {
					auto g = pow((bs::module<T>::parameter_lists()[i]).grad(), 2);
					grad_norm2 += cu_total_sum(g);
				}
				return std::sqrt(grad_norm2);
			}

			void global_g_clipping_by_norm(T max_norm) {
				auto L2_norm = L2_grad_norm();
				if (L2_norm > max_norm) {
					for (int i = 0; i < bs::module<T>::parameter_lists().size(); i++) {
						(bs::module<T>::parameter_lists()[i]).grad() *= (max_norm / L2_norm);
					}
				}
			}

			void Layer_g_clipping_by_norm(T max_norm) {
				for (int i = 0; i < bs::module<T>::Parameter_List().size(); i++) {
					auto L_g = bs::module<T>::Parameter_List()[i].grad();
					auto L_g_L2norm = cu_total_sum(Tpow(L_g, 2));
					if (L_g_L2norm > max_norm)
						bs::module<T>::Parameter_List()[i].grad() *= (max_norm / L_g_L2norm);
				}
			}

		};
	}
}