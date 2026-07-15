# Tensorplusplus
一个带自动求导的c++的神经网络库，使用cuda加速,编译需要调用

h: cu_plusplus.h

lib: cu_plusplus.lib

dll: 
cuda toolkit(cublas64_12.dll cublasLt64_12.dll cudart64_12.dll curand64_10.dll) 

cudnn(cudnn_adv64_9.dll,cudnn_cnn64_9.dll,cudnn_engines_precompiled64_9.dll,cudnn_engines_runtime_compiled64_9.dll,cudnn_engines_tensor_ir64_9.dll,cudnn_ext64_9.dll,cudnn_graph64_9.dll,cudnn_heuristic64_9.dll,cudnn_ops64_9.dll,cudnn64_9.dll)

底层核函数源码cu_plusplus: https://github.com/wz043/cu_plusplus/

使用c++17标准

通过#include "Tensorplusplus.h"开始
