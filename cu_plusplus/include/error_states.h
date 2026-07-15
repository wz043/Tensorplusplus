#pragma once
#include <iostream>

namespace cu_plusplus {
	enum class DType : int {
		Float32 = 0,
		Float64 = 1,
		Float16 = 2,
		BFloat16 = 3
	};

	inline std::ostream& operator<<(std::ostream& os, DType dtype) {
		switch (dtype) {
		case DType::Float32:  os << "float32"; break;
		case DType::Float64:  os << "float64"; break;
		case DType::Float16:  os << "float16"; break;
		case DType::BFloat16: os << "bfloat16"; break;
		default:              os << "unknown"; break;
		}
		return os;
	}

	class Library {
	public:
		Library() = delete;

		static void error_check(int states_code) {
			if (states_code != 0) {
				throw std::runtime_error("Library_fatal_error");
			}
		}

	};

}