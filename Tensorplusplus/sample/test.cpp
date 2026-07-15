// Tensorplusplus.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//
#include "../Tensorplusplus.h"
#include <iostream>
using namespace Tensorplusplus;
namespace tpp = Tensorplusplus;
namespace f = function;

int main()
{
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
	/*Tensor<float> a({1,2,3});
	Tensor<float> b({2,2,3});
	auto c = a + b;
	a.sum_in_dim(1);*/
	
	/*Tensor<float> A({2,3,4});
	A.required_grad() = true;
	auto B = PPO_clip(A,1.0f,-1.0f);
	std::cout << B << std::endl;
	B.backward();
	std::cout << A.grad() << std::endl;

	Tensor<float> X({2,3,4});
	//X.One();
	X.required_grad() = true;
	Tensor<float> Y({2,3,4});
	Y.Zero();
	Y.required_grad() = true;
	auto Z = max(X,Y);
	std::cout << Z << std::endl;
	Z.backward();
	std::cout << X.grad() << std::endl;
	std::cout << Y.grad() << std::endl;*/

	
	auto PPO_loss = loss::PPO<float>(0.5f, 0.0f, {-10.0f,10.0f});
	for (int i = 0; i < 10; i++) {
		Tensor<float> policy({4},Device::CUDA,false);
		policy.required_grad() = true;
		policy.One();
		Tensor<float> value({1}, Device::CUDA, false);
		value.One();
		value.required_grad() = true;
		PPO_loss.record(policy,value,3,1);
		if (i == 9) {
			std::cout << PPO_loss.mean_forward() << std::endl;
			PPO_loss.AGD();
			std::cout << policy.grad() << std::endl;
			std::cout << value.grad() << std::endl;
		}
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
