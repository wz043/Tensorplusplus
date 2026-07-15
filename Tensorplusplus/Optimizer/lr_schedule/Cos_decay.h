#pragma once

constexpr double PI = 3.14159265358979323846;

namespace Tensorplusplus {
	namespace optimizer {
		namespace lr_schedule {
			template<typename T>
			class Cos_decay {
			private:
				double max_lr;
				double min_lr;

				double o;
				double A;

				double bias;
			public:
				int total_step = 0;
				int step_count = 0;

				Cos_decay() = delete;

				Cos_decay(double max_lr, int total_step, double min_lr = double(0)) {
					this->max_lr = max_lr;
					this->min_lr = min_lr;
					this->total_step = total_step;
					this->o = PI / double(total_step);
					this->A = max_lr / 2.0f;
					this->bias = max_lr / 2.0f;
				}


				T Step() {
					double step_lr = A * std::cos(o * step_count) + bias;
					if (step_lr <= this->min_lr || step_count >= total_step)
						return this->min_lr;
					step_count++;
					return T(step_lr);
				}

			};
		
		}
	}
}