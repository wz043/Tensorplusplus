#pragma once

#include "../Tensor/Tensor.h"
#include <filesystem>

namespace Tensorplusplus {
	namespace bs {
		template<typename T>
		class module {
		public:
			explicit module() = delete;

			static inline long long parameter_counts() {
				long long count = 0;
				for (long long i = 0; i < Tensor<T>::parameter_lists.size(); i++) {
					count += Tensor<T>::parameter_lists[i].total_size();
				}
				return count;
			}

			static inline void load_path(std::string path) {
				std::string suffix = path.substr(path.length() - 3);
				if (suffix != "bin") {
					std::cout << "load_path suffix is not bin" << "\n";
					throw std::invalid_argument("load_path suffix is not bin");
				}
				Tensor<T>::load_path = path;
			}

			static inline void save_path(std::string path) {
				Tensor<T>::save_path = path;
			}

			inline static std::vector<Tensor<T>>& parameter_lists() {
				return Tensor<T>::parameter_lists;
			}

			inline static void save_parameters(std::string path) {
				if (path.size() == 0) {
					std::cout << "save_path is null" << std::endl;
					throw std::invalid_argument("save_path is null");
				}
				std::string suffix = path.substr(path.length() - 3);
				if (suffix != "bin") {
					std::cout << "load_path suffix is not bin" << "\n";
					throw std::invalid_argument("load_path suffix is not bin");
				}
				int dotPos = path.find_last_of('.');
				std::string stem = path.substr(0, dotPos);
				std::string ext = path.substr(dotPos);
				int path_counts = 0;
				//std::string newPath = path;
				if (isFileExists_ifstream(path)) {
					while (true) {
						path = stem + "(" + std::to_string(path_counts) + ")" + ext;
						if (!isFileExists_ifstream(path)) {
							std::cout << "save_path is exist, new save_path is " + path << std::endl;
							break;
						}
						path_counts++;
					}
				}
				
				auto file = std::fstream(path, std::ios::out | std::ios::binary);
				for (int i = 0; i < Tensor<T>::parameter_lists.size(); i++) {
					Tensor<T>::parameter_lists[i].save_parameters(file);
				}
				file.close();
				std::cout << path + ":" << "save parameters success" << std::endl;
			}

			inline static void save_parameters() {
				save_parameters(Tensor<T>::save_path);
			}

		private:
			inline static bool isFileExists_ifstream(const std::string& name) {
				return std::filesystem::exists(name) && std::filesystem::is_regular_file(name);
			}
			

		};

	}
}