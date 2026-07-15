// Tensorplusplus.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//
#include "../Tensorplusplus.h"
#include <iostream>
using namespace Tensorplusplus;
namespace tpp = Tensorplusplus;
namespace f = function;


class block1 {
public:
	bs::RMSNorm<float> w0{ 1,{10,1} };
	bs::Linear<float> w1{ 1,1024 };
	bs::Linear<float> w2{ 1,1024 };
	bs::Linear<float> w3{ 1024,1 };

	block1() = default;

	Tensor<float> forward(Tensor<float> x) {
		auto x1 = w0.forward(x);
		auto result = w3.forward(f::silu<float>::forward(w1.forward(x1)) * w2.forward(x1));
		return result;
	}

};

class block2 {
public:
	bs::RMSNorm<float> w0{ 1,{10,1} };
	bs::Linear<float> w1{ 1,1024 };
	bs::Linear<float> w2{ 1,1024 };
	bs::Linear<float> w3{ 1024,1 };

	block2() = default;

	Tensor<float> forward(Tensor<float> x) {
		auto x2 = w0.forward(x);
		auto result = w3.forward(f::silu<float>::forward(w1.forward(x2)) * w2.forward(x2));
		return result;
	}

};

class model {
public:
	block1 B1;
	block2 B2;

	model() = default;

	Tensor<float> forward(Tensor<float>& x) {
		return B2.forward(B1.forward(x));
	}

};

int main()
{
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
    Tensor<float> A({3,2,3});
    A.One();
    Tensor<float> B({6,3,4});
    B.One();
    std::cout << matmul(A, B) << std::endl;
    std::cout << A[2] << std::endl;
    std::cout << "Hello World!\n";


	double max_lr = 3e-4;
	int max_step = 100;

	//double max_y = 5 * max_step + 10;
	//double min_y = 10;

	//double normalized_y = (y - min_y) / (max_y - min_y);

	//bs::module<float>::load_path("test.bin");

	model model1;

	loss::MSE<float> mse_loss;
	optimizer::AdamW<float> Adamw(0.9f, 0.999f, 1e-8, 1e-2);
	//optimizer::Gd<float> Adamw;
	//optimizer::Momentum<float> Adamw(0.9f);
	optimizer::lr_schedule::Cos_decay<float> Linear_decay(max_lr, max_step);
	std::cout << "total_parameter_count" << bs::module<float>::parameter_counts() << std::endl;

	Tensor<float> y({ 10,1 });
	y.One();
	auto y2 = y * 5.0f + 1.0f;
	for (int j = 0; j < max_step; j++) {
		//auto y = i * 5 + 10;
		//double max_y = 5 * max_step + 10;
		//double min_y = 10;
		//double normalized_y = (y - min_y) / (max_y - min_y);

		//A[0] = float(y);
		auto B = model1.forward(y);

		auto C = mse_loss.mean_forward(B, y2);
		std::cout << "mse_loss" << C << std::endl;
		mse_loss.AGD();

		auto lr = Linear_decay.Step();
		//Adamw.global_g_clipping_by_norm(1.0f);
		std::cout << "grad_L2norm" << Adamw.L2_grad_norm() << std::endl;
		Adamw.Update(lr);
		Adamw.Grad_Zero();
		//Tensor<float>::print(model1.B2.w3.bias);
		//Tensor<float>::print(model1.B2.w3.bias.grad());
		/*if (j % 10 == 0)
			i++;*/
	}
	return 0;
}

// 运行程序: Ctrl + F5 或调试 >“开始执行(不调试)”菜单
// 调试程序: F5 或调试 >“开始调试”菜单

// 入门使用技巧: 
//   1. 使用解决方案资源管理器窗口添加/管理文件
//   2. 使用团队资源管理器窗口连接到源代码管理
//   3. 使用输出窗口查看生成输出和其他消息
//   4. 使用错误列表窗口查看错误
//   5. 转到“项目”>“添加新项”以创建新的代码文件，或转到“项目”>“添加现有项”以将现有代码文件添加到项目
//   6. 将来，若要再次打开此项目，请转到“文件”>“打开”>“项目”并选择 .sln 文件
