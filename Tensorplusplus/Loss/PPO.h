#pragma once

#include "loss.h"
#include "../Function/TPP_function.h"

namespace Tensorplusplus {
	namespace loss {
		//adv->GAE: delta(t) = rt + gamma * V(t+1) - V(t) , At = ¡Æn=0->+inf (gamma * lambda)^^n * delta(t+1)
		/*Because the value head predicts the average score that can be obtained in the future under the current state, 
		so here the evaluation of the action's quality is done by the current step score + the predicted score in the future under this action - the predicted score in the future under the current state (since there are both good and bad actions, the delta will not approach 0)*/
		//PPO = -min(ratio * adv, ratio.clip() * adv) + c1 * MSE(value_head - returns) - c2 * softmax(p) * log_softmax(p)
		template<typename T>
		class PPO : public loss<T> {
		private:
			std::vector<Tensor<T>> action_probability;
			std::vector<T> step_reward;
			std::vector<T> done;
			std::vector<Tensor<T>> value_head_predict;
			std::vector<T> prob_clip;
			T gamma;
			T lambda;
			T c1;
			T c2;
		public:
			explicit PPO(T c1,T c2,std::vector<T> prob_clip ,T gamma = T(0.99),T lambda = T(0.95)) {
				this->c1 = c1;
				this->c2 = c2;
				this->gamma = gamma;
				this->lambda = lambda;
				if (prob_clip.size() != 2) {
					std::cout << "PPO_loss error->prob_clip size must be 2 but " << prob_clip.size() << " was received. " << "\n";
					throw std::invalid_argument("PPO_loss error->prob_clip received illegal size");
				}
				if (prob_clip[0] > prob_clip[1]) {
					std::cout << "PPO_loss error->prob_clip up_limit:" << prob_clip[0] << " must more than low_limit:" << prob_clip[1] << "\n";
					throw std::invalid_argument("PPO_loss error->prob_clip up_limit must more than low_limit");
				}
				this->prob_clip = prob_clip;
			}

			void record(Tensor<T> policy_action_probability, Tensor<T> value_head_predict, T step_reward,T done) {
#ifdef _DEBUG
				if (value_head_predict.shape().size() != policy_action_probability.shape().size()) {
					std::cout << "PPO_error:PPO value_head_predict size->" << value_head_predict.shape() << " must equal with policy_head_prob size->"<< policy_action_probability.shape() << "\n";
					throw std::invalid_argument("PPO_error:PPO value_head_predict size must must equal with policy_head_prob size.");
				}

#endif
				this->action_probability.push_back(policy_action_probability);
				this->value_head_predict.push_back(value_head_predict);
				this->step_reward.push_back(step_reward);
				this->done.push_back(done);
			}

			void clear() {
				this->action_probability.clear();
				this->value_head_predict.clear();
				this->step_reward.clear();
				this->done.clear();
			}

			Tensor<T> forward() {
				long long length = step_reward.size() - 1;//Exclude the end
				Tensor<T> adv({length}, action_probability[0].device(),false);
				T GAE = T(0);
				for (long long t = length - 1; t >= 0; t--) {
					T delta = step_reward[t] + gamma * value_head_predict[t + 1][0] * (1 - done[t]) - value_head_predict[t][0];
					GAE = delta + gamma * lambda * (1 - done[t]) * GAE;
					adv[t] = GAE;
				}
				value_head_predict.pop_back();
				auto Tvalue = Batch_cat(value_head_predict);
				auto T_value = Tvalue.squeeze(Tvalue.shape().size() - 1);
				Tensor<T> returns = Add_op<T>::add(adv ,T_value);
				std::vector<Tensor<T>> ratio(action_probability.size() - 1);
				ratio[0] = action_probability[0];
				//T eps = T(1e-8);
				for (long long i = 1; i < action_probability.size() - 1; i++) {
					ratio[i] = exp(ln(action_probability[i]) - ln(action_probability[i - 1]));
				}
				auto T_ratio = Batch_cat(ratio);
				
				if (c2 == T(0)) {
					this->loss_tensor = -min(T_ratio.sum_in_dim(T_ratio.shape().size() - 1) * adv, PPO_clip(T_ratio.sum_in_dim(T_ratio.shape().size() - 1), prob_clip[0], prob_clip[1]) * adv) + c1 * pow((returns - T_value), 2);
				}
				else {
					auto T_police = Batch_cat(action_probability);
					this->loss_tensor = -min(T_ratio.sum_in_dim(T_ratio.shape().size() - 1) * adv, PPO_clip(T_ratio.sum_in_dim(T_ratio.shape().size() - 1), prob_clip[0], prob_clip[1]) * adv) + c1 * pow((returns - T_value), 2) - c2 * T_police * ln(T_police);
				}

				return this->loss_tensor;
			}

			T mean_forward() {
				auto forward_tensor = forward();
				this->loss_tensor = forward_tensor / T(forward_tensor.total_size());

				T result = cu_total_sum(this->loss_tensor) / T(this->loss_tensor.total_size());
				return result;
			}

			T sum_forward() {
				this->loss_tensor = forward();

				T result = cu_total_sum(this->loss_tensor);
				return result;
			}

		};
	}
}