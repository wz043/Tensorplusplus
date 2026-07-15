#pragma once

#include <cmath>

#include "optimizer.h"


namespace Tensorplusplus {
	namespace optimizer{
		template<typename T>
		class Gd : public optimizer<T>{
		public:
			explicit Gd() = default;

			void Update(T lr) override {
				for (int i = 0; i < bs::module<T>::parameter_lists().size(); i++) {
					if (bs::module<T>::parameter_lists()[i].required_grad())
						bs::module<T>::parameter_lists()[i] -= (lr * (bs::module<T>::parameter_lists()[i]).grad());
				}
			}

		};
	}
}
