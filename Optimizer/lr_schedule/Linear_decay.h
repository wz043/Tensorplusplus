#pragma once

namespace Tensorplusplus {
	namespace optimizer {
		namespace lr_schedule {
			template<typename T>
			class Linear_decay {
			private:
				double max_lr;
				double min_lr;
				
				double k;
				double b;

			public:
				int total_step = 0;
				int step_count = 0;

				Linear_decay() = delete;

				Linear_decay(double max_lr, int total_step, double min_lr = double(0)) {
					this->max_lr = max_lr;
					this->min_lr = min_lr;
					this->total_step = total_step;
					this->k = -max_lr / double(total_step);
					this->b = max_lr;
				}

				T Step() {
					double step_lr = k * step_count + b;
					if (step_lr <= this->min_lr || step_count >= total_step)
						return this->min_lr;
					step_count++;
					return T(step_lr);
				}

			};
		}
	}
}
