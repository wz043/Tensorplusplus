#pragma once
#pragma comment(lib, "cu_plusplus.lib")
#include "cu_plusplus/include/cu_plusplus.h"
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <cstdlib>
#include "../Resource/Resource.h"
#include "Tensor_op.h"

//DEFINE

namespace cupp =  cu_plusplus;

namespace Tensorplusplus {

	using biovault::bfloat16_t;
	using half_float::half;

	enum class Device : int {
		CPU = 0,
		CUDA = 1,
	};

	template<typename T>
	class Tensor_op;

	template<typename T>
	class Block;

	template<typename T>
	class Tensor {
	public:
		explicit Tensor() = default;

		explicit Tensor(std::vector<long long> shape_input, Device device = Device::CUDA, bool need_init = true, bool is_parameter = false, bool required_grad = false, long long random_seed = 42) {
			float E = 0.0f;
			long long max_size = 0;
			for (int i = 0; i < shape_input.size(); i++) {
				if(shape_input[i] > max_size)
					max_size = shape_input[i];
			}
			float S = 20.0f / (float)max_size;
			
			block = new Block<T>(shape_input, device, need_init, is_parameter, required_grad, E, S, random_seed);
			//Register_parameter
			if (is_parameter) {
				parameter_lists.push_back(*this);
			}
		}

		explicit Tensor(std::vector<long long> shape_input, float E, float S, long long random_seed, Device device = Device::CUDA, bool need_init = true, bool is_parameter = false, bool required_grad = false) {
			block = new Block<T>(shape_input, device, need_init, is_parameter, required_grad, E, S, random_seed);
			//Register_parameter
			if (is_parameter) {
				parameter_lists.push_back(*this);
			}
		}

		inline static std::vector<Tensor<T>> parameter_lists;
		inline static std::string save_path; 
		inline static std::string load_path;

		void backward() {
			block_init_check();
			if (this->required_grad() == false)
				return;

			Tensor<T> grad_in(this->shape(), this->device(), false);
			grad_in.One();

			if (this->grad().initialize() == false) {
				this->grad() = Tensor<T>(this->shape(), this->device(), false);
			}

			this->grad() += grad_in;

			if (this->op() != nullptr) {
				std::vector<Tensor<T>> grad_out = this->op()->backward(this->tensor_from(), grad_in);

				for (int i = 0; i < this->tensor_from().size(); i++) {
					tensor_from()[i].backward(grad_out[i]);
				}

				this->tensor_from().clear();

			}
			this->used_count() = 0;
			this->grad_in().clear();
		}

		void backward(Tensor<T>& grad) {
			block_init_check();
			if (this->required_grad() == false) 
				return;

			this->grad_in().push_back(grad);
			if (this->grad_in().size() != this->used_count())
				return;

			Tensor<T> grad_in(this->shape(), this->device(), false);
			for (int i = 0; i < this->grad_in().size(); i++) {
				grad_in += this->grad_in()[i];
			}

			if (this->grad().initialize() == false) {
				this->grad() = Tensor<T>(this->shape(), this->device(), false);
			}

			this->grad() += grad_in;

			if (this->op() != nullptr) {
				std::vector<Tensor<T>> grad_out = this->op()->backward(this->tensor_from(), grad_in);

				for (int i = 0; i < this->tensor_from().size(); i++) {
					tensor_from()[i].backward(grad_out[i]);
				}

				this->tensor_from().clear();

			}
			this->used_count() = 0;
			this->grad_in().clear();
		}

		void block_init_check() {
#ifdef _DEBUG
			if (block == nullptr) {
				std::cout << "Tensor error: Tensor block not init" << "\n";
				throw std::invalid_argument("Tensor error:Tensor block not init");
			}
#endif
		}

		Tensor<T> reshape(std::vector<long long> reshape) {
			block_init_check();
			Reshape_op<T>* reshape_op = new Reshape_op<T>();
			Tensor<T> B = reshape_op->forward(*this, reshape);
			return B;
		}

		Tensor<T> sum_in_dim(long long dim) {
			block_init_check();
			Sum_in_dim_op<T>* sum_in_dim_op = new Sum_in_dim_op<T>();
			Tensor<T> B = sum_in_dim_op->forward(*this, dim);
			return B;
		}

		Tensor<T> broadcast_in_dim(long long dim, long long up_dim_shape) {
			block_init_check();
			Broadcast_in_dim_op<T>* broadcast_in_dim = new Broadcast_in_dim_op<T>();
			auto B = broadcast_in_dim->forward(*this, dim, up_dim_shape);
			return B;
		}

		T* View() {
			block_init_check();
			return block->memory;
		}

		T& operator[](long long index) {
#ifdef _DEBUG
			if (index >= block->total_size || index < 0) {
				std::cout << "Tensor_error Index_out_of_range: " << index << std::endl;
				throw std::invalid_argument("index of range");
			}
#endif
			return block->memory[index];
		}


		Tensor<T> slice(long long dim, std::vector<long long> slice) {
			block_init_check();
#ifdef _DEBUG
			if (slice.size() != 3 && slice.size() != 2) {
				std::cout << "Tensor_error->Slice:" << slice << " slice should have 2 or 3 elements" << "\n";
				throw std::invalid_argument("Tensor_error->Slice: slice should have 2 or 3 elements");
			}
#endif
			Slice_op<T>* slice_op = new Slice_op<T>();
			Tensor<T> B;
			if (slice.size() == 3)
				B = slice_op->forward(*this, dim, slice[0], slice[1], slice[2]);
			else if (slice.size() == 2)
				B = slice_op->forward(*this, dim, slice[0], slice[1], 1);
			return B;
		}

		Tensor<T> operator-() {
			block_init_check();
			Opposite_op<T>* opposite_op = new Opposite_op<T>();
			auto B = opposite_op->forward(*this);
			return B;
		}

		Tensor<T>& operator+=(Tensor<T> other) {
			block_init_check();
			ref_add<T>(*this, other);
			return *this;
		}


		Tensor<T>& operator-=(Tensor<T> other) {
			block_init_check();
			ref_sub<T>(*this, other);
			return *this;
		}

		Tensor<T>& operator*=(Tensor<T> other) {
			block_init_check();
			ref_hadamard<T>(*this, other);
			return *this;
		}


		Tensor<T>& operator/=(Tensor<T> other) {
			block_init_check();
			ref_division<T>(*this, other);
			return *this;
		}


		Tensor<T> transpose() {
			block_init_check();
			Transpose_op<T>* transpose_op = new Transpose_op<T>();
			auto B = transpose_op->forward(*this);
			return B;
		}

		Tensor<T> transpose(long long dim1, long long dim2) {
			block_init_check();
			Transpose_with_dim_op<T>* transpose_with_dim_op = new Transpose_with_dim_op<T>();
			auto B = transpose_with_dim_op->forward(*this, dim1, dim2);
			return B;
		}

		Tensor<T> pow(int pow_number) {
			block_init_check();
			Pow_op<T>* pow_op = new Pow_op<T>();
			auto C = pow_op->forward(*this, pow_number);
			return C;
		}

		Tensor<T> sqrt() {
			block_init_check();
			Sqrt_op<T>* sqrt_op = new Sqrt_op<T>();
			auto B = sqrt_op->forward(*this);
			return B;
		}

		Tensor<T> exp() {
			block_init_check();
			Exp_op<T>* exp_op = new Exp_op<T>();
			auto B = exp_op->forward(*this);
			return B;
		}

		Tensor<T> ln() {
			block_init_check();
			Ln_op<T>* ln_op = new Ln_op<T>();
			auto B = ln_op->forward(*this);
			return B;
		}

		Tensor<T> cos() {
			block_init_check();
			Cos_op<T>* cos_op = new Cos_op<T>();
			auto B = cos_op->forward(*this);
			return B;
		}

		Tensor<T> sin() {
			block_init_check();
			Sin_op<T>* sin_op = new Sin_op<T>();
			auto B = sin_op->forward(*this);
			return B;
		}

		Tensor<T> squeeze() {
			block_init_check();
			Squeeze_op<T>* squeeze_op = new Squeeze_op<T>();
			auto B = squeeze_op->forward(*this);
			return B;
		}

		Tensor<T> squeeze(long long squeeze_dim) {
			block_init_check();
			Squeeze_with_dim_op<T>* squeeze_with_dim_op = new Squeeze_with_dim_op<T>();
			auto B = squeeze_with_dim_op->forward(*this,squeeze_dim);
			return B;
		}

		T total_sum() {
			block_init_check();
			return cu_total_sum<T>(*this);
		}

		friend std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
			if (tensor.block == nullptr) {
				os << "Tensor(uninitialized)";
				return os;
			}

			os << "{Tensor_data:{";
			if (tensor.block->total_size <= 100) {

				for (long long i = 0; i < tensor.block->total_size; i++) {
					if (i == 0)
						os << tensor.block->memory[i];
					else
						os << "," << tensor.block->memory[i];
				}
			}
			else {

				for (long long i = 0; i < 50; i++) {
					if (i == 0)
						os << tensor.block->memory[i];
					else
						os << "," << tensor.block->memory[i];
				}

				for (long long i = tensor.block->total_size - 50; i < tensor.block->total_size; i++) {
					if (i == tensor.block->total_size - 50)
						os << " ...... " << tensor.block->memory[i];
					else
						os << "," << tensor.block->memory[i];
				}
			}

			os << "}" << "," << "Tensor_device:";
			switch (tensor.block->device) {
			case Device::CPU:
				os << "CPU";
				break;
			case Device::CUDA:
				os << "CUDA";
				break;
			default:
				os << "unknown";
			}
			os << "," << "Tensor_dtype:";
			switch (tensor.block->data_type) {
			case cupp::DType::Float32:
				os << "float32";
				break;
			case cupp::DType::Float64:
				os << "float64";
				break;
			case cupp::DType::Float16:
				os << "float16";
				break;
			case cupp::DType::BFloat16:
				os << "bfloat16";
				break;
			default:
				os << "unknown";
			}

			os << "," << "Tensor_shape:{";
			for (long long i = 0; i < (long long)tensor.block->shape.size(); i++) {
				if (i == 0)
					os << tensor.block->shape[i];
				else
					os << "," << tensor.block->shape[i];
			}
			os << "}";
			os << "," << "total_size:" << tensor.block->total_size;
			os << "," << "required_grad:" << tensor.block->required_grad << "}";

			return os;
		}

		bool initialize() {
			return block;
		}

		const std::vector<long long>& shape(){
			block_init_check();
			return block->shape;
		}

		const long long& total_size() {
			block_init_check();
			return block->total_size;
		}
		
		const long long& batch() {
			block_init_check();
			return block->batch;
		}

		const long long& matmul_batch() {
			block_init_check();
			return block->matmul_batch;
		}

		const long long& conv2d_batch() {
			block_init_check();
			return block->conv2d_batch;
		}

		const cupp::DType& dtype() {
			block_init_check();
			return block->data_type;
		}

		const Device& device() {
			block_init_check();
			return block->device;
		}

		const bool& is_parameter() {
			block_init_check();
			return block->is_parameter;
		}

		std::vector<Tensor<T>>& tensor_from() {
			block_init_check();
			return block->tensor_from;
		}

		std::vector<Tensor<T>>& grad_in() {
			block_init_check();
			return block->grad_in;
		}

		int& used_count() {
			block_init_check();
			return block->used_count;
		}

		const int& heap_used_count() {
			block_init_check();
			return block->heap_used_count;
		}

		Tensor_op<T>*& op() {
			block_init_check();
			return block->op;
		}

		bool& required_grad() {
			block_init_check();
			return block->required_grad;
		}

		Tensor<T>& grad() {
			block_init_check();
			return block->grad;
		}

		//Zero
		void Zero() {
			block_init_check();
			cupp::Library::error_check(cupp::cu_Zero(block->memory, block->total_size, block->data_type));
		}

		//One
		void One() {
			block_init_check();
			cupp::Library::error_check(cupp::cu_One(block->memory, block->total_size, block->data_type));
		}

		Tensor<T> Copy() {
			block_init_check();
			Tensor<T> B(block->shape, block->device, false);
			cupp::Library::error_check(cupp::cu_Copy(block->memory, B.View(), block->total_size, block->data_type));
			return B;
		}

		Tensor(const Tensor<T>& other) : block(other.block) {
			if (initialize())
				block->heap_used_count++;
		}

		Tensor<T>& operator=(const Tensor<T>& other) {
			if (this == &other || this->block == other.block) {
				return *this;
			}
			Dispose();
			block = other.block;
			block->heap_used_count++;
			return *this;
		}

		Tensor(const Tensor<T>&& other) noexcept : block(other.block){
			if (initialize())
				block->heap_used_count++;
		}

		Tensor<T>& operator=(const Tensor<T>&& other) noexcept {
			if (this == &other || this->block == other.block) {
				return *this;
			}
			Dispose();
			block = other.block;
			block->heap_used_count++;
			return *this;
		}

		void Dispose() {
			//std::cout << block->heap_used_count << std::endl;
			if (block == nullptr)
				return;
			if (block->heap_used_count == 1) {
				if (block->memory != nullptr) {
					//std::cout << "cuda free" <<std::endl;
					void* temp = block->memory;
					cupp::Library::error_check(cupp::cu_memory_free(temp));
					block->memory = nullptr;
				}

				if (block->op != nullptr) {
					//std::cout << "op free" << std::endl;
					delete block->op;
				}
				//std::cout << "block free" << std::endl;
				delete block;
			}
			else
				block->heap_used_count--;
		}

		~Tensor() { Dispose(); }

		inline void save_parameters(std::fstream& file) {
			block_init_check();
			file.write(
				reinterpret_cast<const char*>(block->memory),
				sizeof(T) * block->total_size
			);
			
			if (file.fail()) {
				std::cerr << "Tensor_save_weight failed" << std::endl;
				throw std::runtime_error("Tensor_save_weight");
			}

		}

	private:
		template<typename T>
		class Block {
		public:
			T* memory = nullptr;
			std::vector<long long> shape;

			Device device;
			cupp::DType data_type;
			long long total_size = 0;
			long long batch;
			long long matmul_batch;
			long long conv2d_batch;

			std::vector<Tensor<T>> tensor_from;

			std::vector<Tensor<T>> grad_in;
			int used_count = 0;
			Tensor_op<T>* op = nullptr;

			bool is_parameter = false;
			Tensor<T> grad;
			bool required_grad = false;

			int heap_used_count = 1;

			//init prevent multiple load_file
			struct load_file {
				std::ifstream inFile;
				bool init = false;
			};

			static inline load_file loader;

			explicit Block() = default;

			explicit Block(std::vector<long long> shape_input, Device device = Device::CUDA, bool need_init = true, bool is_parameter = false, bool required_grad = false, float E = 0.0f, float S = 1.0f, long long random_seed = 42) {
#ifdef _DEBUG
				if (device != Device::CUDA) {
					std::cout << "Tensor error:device_not_support" << "\n";
					throw std::invalid_argument("Tensor error:device_not_support");
				}
#endif
				this->device = device;

				if constexpr (std::is_same_v<T, float>) {
					this->data_type = cupp::DType::Float32;
				}
				else if constexpr (std::is_same_v<T, double>) {
					this->data_type = cupp::DType::Float64;
				}
				else if constexpr (std::is_same_v<T, half>) {
					this->data_type = cupp::DType::Float16;
				}
				else if constexpr (std::is_same_v<T, bfloat16_t>) {
					this->data_type = cupp::DType::BFloat16;
				}

				shape = shape_input;

				total_size = 1;
				for (long long i = 0; i < (long long)shape.size(); i++) {
#ifdef _DEBUG
					if (shape[i] <= 0) {
						std::cout << "Tensor error:error_input_shape:" << shape << "\n";
						throw std::invalid_argument("error_shape_input");
					}
#endif
					total_size *= shape[i];
				}

				batch = shape_input[0];
				matmul_batch = 1;
				conv2d_batch = 1;
				if (shape_input.size() > 2)
				{
					for (long long i = 0; i < (long long)shape_input.size() - 2; i++)
					{
						matmul_batch *= shape_input[i];
						if (i < (long long)shape_input.size() - 3)
							conv2d_batch *= shape_input[i];
					}
				}

				void* temp = memory;
				cupp::Library::error_check(cupp::cu_memory_apply(temp, total_size, data_type));
				memory = (T*)temp;


				//weight_load
				if (is_parameter && load_path.size() != 0) {
					if (loader.init == false) {
						loader.inFile.open(load_path, std::ios::binary);
						loader.init = true;
					}
					if (!loader.inFile.is_open()) {
						std::cout << "weight_file_not_found" << std::endl;
						throw std::invalid_argument("weight_file_not_found");
					}

					loader.inFile.read(reinterpret_cast<char*>(this->View()), sizeof(T) * total_size);

					if (loader.inFile.gcount() != sizeof(T) * total_size) {
						std::cout << "read_error" << std::endl;
						throw std::runtime_error("incomplete read");
					}
					if (loader.inFile.fail()) {
						std::cerr << "read failed" << std::endl;
						throw std::runtime_error("read failed");
					}

				}
				else {
					if (need_init) {
						//cu_rand_init
						cupp::Library::error_check(cupp::cu_rand_init(memory, total_size, E, S, random_seed, data_type));
					}
					else {
						//Zero
						cupp::Library::error_check(cupp::cu_Zero(memory, total_size, data_type));
					}
				}

				this->required_grad = required_grad;
			}

			void* View() { return memory; }

			Block(const Block& other) = delete;

			Block& operator=(const Block& other) = delete;

			Block(Block&& other) noexcept = delete;

			Block& operator=(Block&& other) noexcept = delete;

		};

		Block<T>* block = nullptr;

	};

	


	//Tensor_op
	//add
	template<typename T>
	Tensor<T> operator+(Tensor<T> A, Tensor<T> B) {
		Add_op<T>* add_op = new Add_op<T>();
		auto C = add_op->forward(A, B);
		return C;
	}

	template<typename T>
	Tensor<T> operator+(Tensor<T> A, T B) {
		Add_with_const_op<T>* add_with_const_op = new Add_with_const_op<T>();
		auto C = add_with_const_op->forward(A, B);
		return C;
	}

	template<typename T>
	Tensor<T> operator+(T A, Tensor<T> B) {
		Add_with_const_op<T>* add_with_const_op = new Add_with_const_op<T>();
		auto C = add_with_const_op->forward(B, A);
		return C;
	}

	//sub
	template<typename T>
	Tensor<T> operator-(Tensor<T> A, Tensor<T> B) {
		Sub_op<T>* sub_op = new Sub_op<T>();
		auto C = sub_op->forward(A, B);
		return C;
	}

	template<typename T>
	Tensor<T> operator-(Tensor<T> A, T B) {
		Sub_with_rconst_op<T>* sub_with_Rconst_op = new Sub_with_rconst_op<T>();
		auto C = sub_with_Rconst_op->forward(A, B);
		return C;
	}

	template<typename T>
	Tensor<T> operator-(T A, Tensor<T> B) {
		Sub_with_lconst_op<T>* sub_with_Lconst_op = new Sub_with_lconst_op<T>();
		auto C = sub_with_Lconst_op->forward(A, B);
		return C;
	}

	//matmul
	template<typename T>
	Tensor<T> matmul(Tensor<T> A, Tensor<T> B) {
		Matmul_op<T>* matmul_op = new Matmul_op<T>();
		auto C = matmul_op->forward(A, B);
		return C;
	}

	//Hadamard
	template<typename T>
	Tensor<T> operator*(Tensor<T> A, Tensor<T> B) {
		Hadamard_op<T>* hadamard_op = new Hadamard_op<T>();
		auto C = hadamard_op->forward(A, B);
		return C;
	}

	//mul_with_const
	template<typename T>
	Tensor<T> operator*(Tensor<T> A, T b) {
		Hadamard_with_const_op<T>* hadamard_with_const_op = new Hadamard_with_const_op<T>();
		auto C = hadamard_with_const_op->forward(A, b);
		return C;
	}

	template<typename T>
	Tensor<T> operator*(T a, Tensor<T> B) {
		Hadamard_with_const_op<T>* hadamard_with_const_op = new Hadamard_with_const_op<T>();
		auto C = hadamard_with_const_op->forward(B, a);
		return C;
	}

	//Pow
	template<typename T>
	Tensor<T> pow(Tensor<T> A, int b) {
		Pow_op<T>* pow_op = new Pow_op<T>();
		auto C = pow_op->forward(A, b);
		return C;
	}

	//Division
	template<typename T>
	Tensor<T> operator/(Tensor<T> A, Tensor<T> B) {
		Division_op<T>* division_op = new Division_op<T>();
		auto C = division_op->forward(A, B);
		return C;
	}

	template<typename T>
	Tensor<T> operator/(T a, Tensor<T> B) {
		Division_with_lconst_op<T>* div_with_Lconst_op = new Division_with_lconst_op<T>();
		auto C = div_with_Lconst_op->forward(a, B);
		return C;
	}

	template<typename T>
	Tensor<T> operator/(Tensor<T> A, T b) {
		Division_with_rconst_op<T>* div_with_Rconst_op = new Division_with_rconst_op<T>();
		auto C = div_with_Rconst_op->forward(A, b);
		return C;
	}

	//exp
	template<typename T>
	Tensor<T> exp(Tensor<T> A) {
		Exp_op<T>* exp_op = new Exp_op<T>();
		auto B = exp_op->forward(A);
		return B;
	}

	//ln
	template<typename T>
	Tensor<T> ln(Tensor<T> A) {
		Ln_op<T>* ln_op = new Ln_op<T>();
		auto B = ln_op->forward(A);
		return B;
	}

	//sqrt
	template<typename T>
	Tensor<T> sqrt(Tensor<T> A) {
		Sqrt_op<T>* sqrt_op = new Sqrt_op<T>();
		auto B = sqrt_op->forward(A);
		return B;
	}

	//cos
	template<typename T>
	Tensor<T> cos(Tensor<T> A) {
		Cos_op<T>* cos_op = new Cos_op<T>();
		auto B = cos_op->forward(A);
		return B;
	}

	//sin
	template<typename T>
	Tensor<T> sin(Tensor<T> A) {
		Sin_op<T>* sin_op = new Sin_op<T>();
		auto B = sin_op->forward(A);
		return B;
	}


	//Embedding
	template<typename T>
	Tensor<T> Embedding(Tensor<T> A, Tensor<T> vocab);

	//Conv2D
	template<typename T>
	Tensor<T> conv2d(Tensor<T> A,Tensor<T> kernel,std::vector<long long> stride, std::vector<long long> padding) {
		Conv2D_op<T>* conv2d_op = new Conv2D_op<T>();
		auto B = conv2d_op->forward(A,kernel,stride,padding);
		return B;
	}

	template<typename T>
	Tensor<T> Batch_cat(std::vector<Tensor<T>> tensor_list) {
		Batch_cat_op<T>* batch_cat_op = new Batch_cat_op<T>();
		auto B = batch_cat_op->forward(tensor_list);
		return B;
	}

	template<typename T>
	Tensor<T> PPO_clip(Tensor<T> A,T down_limit ,T up_limit) {
		PPO_clip_op<T>* ppo_clip_op = new PPO_clip_op<T>();
		auto B = ppo_clip_op->forward(A,down_limit,up_limit);
		return B;
	}

	template<typename T>
	Tensor<T> min(Tensor<T> A,Tensor<T> B) {
		Min_op<T>* min_op = new Min_op<T>();
		auto C = min_op->forward(A,B);
		return C;
	}

	template<typename T>
	Tensor<T> max(Tensor<T> A, Tensor<T> B) {
		Max_op<T>* max_op = new Max_op<T>();
		auto C = max_op->forward(A, B);
		return C;
	}

}