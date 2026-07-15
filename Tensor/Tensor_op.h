#pragma once

#include <iostream>
#include <string>
#include <vector>
#include "cu_plusplus/include/cu_plusplus.h"
#include "../Resource/Resource.h"
/*

                                A
                                |
                                |
            [y = f(x)]		g * f'(x)		Chain rule
                                A
                                |
                            g	|
                                |
*/

namespace cupp = cu_plusplus;

namespace Tensorplusplus {
    template<typename T>
    class Tensor;

    template<typename T>
    std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec) {
        os << "{";
        for (size_t i = 0; i < vec.size(); i++) {
            if (i != 0) os << ", ";
            os << vec[i];
        }
        os << "}";
        return os;
    }

    template<typename T>
    class Tensor_op {
    public:
        Tensor_op() = default;

        virtual std::vector<Tensor<T>> backward(std::vector<Tensor<T>>& tensor_from, Tensor<T>& grad_in) = 0;

        virtual ~Tensor_op() = default;
    };

    inline std::vector<long long> Broadcast_rule(std::vector<long long> A_shape, std::vector<long long> B_shape, int tail_ignore, std::string op_with_function_name) {
        long long max_length = A_shape.size() >= B_shape.size() ? (long long)A_shape.size() : (long long)B_shape.size();
        long long min_length = A_shape.size() < B_shape.size() ? (long long)A_shape.size() : (long long)B_shape.size();
        std::vector<long long> max_length_shape = A_shape.size() >= B_shape.size() ? A_shape : B_shape;
        std::vector<long long> min_length_shape = A_shape.size() < B_shape.size() ? A_shape : B_shape;

        std::vector<long long> filled_shape(max_length,1);
        for (long long i = 0; i < max_length - tail_ignore; i++)
        {
            if (i >= max_length - min_length)
                filled_shape[i] = min_length_shape[i - (max_length - min_length)];
        }
#ifdef _DEBUG
        if (A_shape.size() > B_shape.size())
        {
            for (int i = 0; i < max_length - tail_ignore; i++)
            {
                if (A_shape[i] % filled_shape[i] != 0)
                {
                    std::cout << op_with_function_name << "The shape of B : " << B_shape << " can't broadcast to A: " << A_shape << "\n";
                    throw std::invalid_argument("The shape of B can't broadcast to A");
                }
                if ((A_shape[i] == filled_shape[i]) && i != 0)
                {
                    for (int j = i + 1; j < max_length - tail_ignore; j++)
                    {
                        if (A_shape[j] != filled_shape[j])
                        {
                            std::cout << op_with_function_name << "The shape of B : " << B_shape << " can't broadcast to A: " << A_shape << "\n";
                            throw std::invalid_argument("The shape of B can't broadcast to A");
                        }
                    }
                    break;
                }
            }
        }
        else if (B_shape.size() > A_shape.size())
        {
            for (int i = 0; i < max_length - 1 - tail_ignore; i++)
            {
                if (B_shape[i] % filled_shape[i] != 0)
                {
                    std::cout << op_with_function_name << "The shape of A : " << A_shape << " can't broadcast to B: " << B_shape << "\n";
                    throw std::invalid_argument("The shape of A can't broadcast to B");
                }
                if ((B_shape[i] == filled_shape[i]) && i != 0)
                {
                    for (int j = i + 1; j < max_length - tail_ignore; j++)
                    {
                        if (B_shape[j] != filled_shape[j])
                        {
                            std::cout << op_with_function_name << "The shape of A : " << A_shape << " can't broadcast to B: " << B_shape << "\n";
                            throw std::invalid_argument("The shape of A can't broadcast to B");
                        }
                    }
                    break;
                }
            }
        }
        else
        {
            for (int i = 0; i < max_length - tail_ignore; i++)
            {
                if (B_shape[i] % A_shape[i] != 0 && A_shape[i] % B_shape[i] != 0)
                {
                    std::cout << op_with_function_name << "The shape of A : " << A_shape << " can't broadcast to B: " << B_shape << "\n";
                    throw std::invalid_argument("The shape of A can't broadcast to B");
                }
                if (B_shape[i] == A_shape[i])
                {
                    for (int j = i + 1; j < max_length - tail_ignore; j++)
                    {
                        if ((B_shape[j] != A_shape[j]) && i != 0)
                        {
                            std::cout << op_with_function_name << "The shape of A : " << A_shape << " can't broadcast to B: " << B_shape << "\n";
                            throw std::invalid_argument("The shape of A can't broadcast to B");
                        }
                    }
                    break;
                }
            }
        }
#endif
        std::vector<long long> C_shape(max_length,0);
        for (long i = 0; i < C_shape.size() - tail_ignore; i++)
        {
            C_shape[i] = max_length_shape[i] > filled_shape[i] ? max_length_shape[i] : filled_shape[i];
        }

        return C_shape;
    }

    //A_to_B
    template<typename T>
    inline Tensor<T> batch_down(Tensor<T> A, Tensor<T> B) {
#ifdef _DEBUG
        if (A.total_size() < B.total_size())
        {
            std::cout << "Tensor_operation_error:batch_down_op-> A total size" << A.total_size() << " less than B total size" << B.total_size() << "\n";
            throw std::invalid_argument("Tensor_operation_error:batch_down_op-> A total size less than B total size");
        }
#endif

        Tensor<T> C(B.shape(), B.device(), false);
        cupp::Library::error_check(cupp::cu_batch_down(A.View(), C.View(), A.total_size(), B.total_size(), A.dtype()));
        return C;
    }

    template<typename T>
    class Add_op : public Tensor_op<T> {
    public:
        Add_op() = default;

        Tensor<T> forward(Tensor<T>& A, Tensor<T>& B) {
            Tensor<T> C = add(A, B);

            A.used_count()++;
            B.used_count()++;

            C.required_grad() = true;
            C.tensor_from().push_back(A);
            C.tensor_from().push_back(B);
            C.op() = this;
            return C;
        }

        std::vector<Tensor<T>> backward(std::vector<Tensor<T>>& tensor_from, Tensor<T>& grad_in) override {
            std::vector<Tensor<T>> grad_out;
            grad_out.push_back(batch_down(grad_in, tensor_from[0]));
            grad_out.push_back(batch_down(grad_in, tensor_from[1]));
            return grad_out;
        }


        Tensor<T> static add(Tensor<T> A, Tensor<T> B) {
            std::vector<long long> C_shape = Broadcast_rule(A.shape(), B.shape(), 0, "Add_op->");
            Tensor<T> C(C_shape, A.device(), false);
            cupp::Library::error_check(cupp::cu_add(A.View(), B.View(), C.View(), A.total_size(), B.total_size(), A.dtype()));
            return C;

        }
    };

    template<typename T>
    class Add_with_const_op : public Tensor_op<T> {
    public:
        Add_with_const_op() = default;

        Tensor<T> forward(Tensor<T>& A, T& b) {
            Tensor<T> C = add_with_const(A, b);

            A.used_count()++;

            C.required_grad() = true;
            C.tensor_from().push_back(A);
            C.op() = this;
            return C;
        }

        std::vector<Tensor<T>> backward(std::vector<Tensor<T>>& tensor_from, Tensor<T>& grad_in) override {
            std::vector<Tensor<T>> grad_out;
            grad_out.push_back(grad_in);
            return grad_out;
        }

        Tensor<T> static add_with_const(Tensor<T> A, T b) {
            Tensor<T> C(A.shape(), A.device(), false);
            cupp::Library::error_check(cupp::cu_add_with_const(A.View(), &b, C.View(), A.total_size(), A.dtype()));
            return C;
        }

    };

    template<typename T>
    class Opposite_op : public Tensor_op<T> {
    public:
        Opposite_op() = default;

        Tensor<T> forward(Tensor<T>& A) {
            Tensor<T> B = opposite(A);

            A.used_count()++;

            B.required_grad() = true;
            B.tensor_from().push_back(A);
            B.op() = this;
            return B;
        }

        std::vector<Tensor<T>> backward(std::vector<Tensor<T>>& tensor_from, Tensor<T>& grad_in) override {
            std::vector<Tensor<T>> grad_out;
            grad_out.push_back(opposite(grad_in));
            return grad_out;
        }

        Tensor<T> static opposite(Tensor<T> A) {
            Tensor<T> B(A.shape(), A.device(), false);
            cupp::Library::error_check(cupp::cu_opposite(A.View(), B.View(), A.total_size(), A.dtype()));
            return B;
        }

    };

    template<typename T>
    class Sub_op : public Tensor_op<T> {
    public:
        Sub_op() = default;

        Tensor<T> forward(Tensor<T>& A, Tensor<T>& B) {
            Tensor<T> C = sub(A, B);

            A.used_count()++;
            B.used_count()++;

            C.required_grad() = true;
            C.tensor_from().push_back(A);
            C.tensor_from().push_back(B);
            C.op() = this;
            return C;
        }

        std::vector<Tensor<T>> backward(std::vector<Tensor<T>>& tensor_from, Tensor<T>& grad_in) override {
            std::vector<Tensor<T>> grad_out;
            grad_out.push_back(batch_down(grad_in, tensor_from[0]));
            grad_out.push_back(batch_down(Opposite_op<T>::opposite(grad_in), tensor_from[1]));
            return grad_out;
        }


        Tensor<T> static sub(Tensor<T> A, Tensor<T> B) {
            std::vector<long long> C_shape = Broadcast_rule(A.shape(), B.shape(), 0, "Sub_op->");
            Tensor<T> C(C_shape, A.device(), false);
            cupp::Library::error_check(cupp::cu_sub(A.View(), B.View(), C.View(), A.total_size(), B.total_size(), A.dtype()));
            return C;
        }

    };

    template<typename T>
    class Sub_with_rconst_op : public Tensor_op<T> {
    public:
        Sub_with_rconst_op() = default;

        Tensor<T> forward(Tensor<T>& A, T& b) {
            Tensor<T> C = sub_with_rconst(A, b);

            A.used_count()++;

            C.required_grad() = true;
            C.tensor_from().push_back(A);
            C.op() = this;
            return C;
        }

        std::vector<Tensor<T>> backward(std::vector<Tensor<T>>& tensor_from, Tensor<T>& grad_in) override {
            std::vector<Tensor<T>> grad_out;
            grad_out.push_back(grad_in);
            return grad_out;
        }

        Tensor<T> static sub_with_rconst(Tensor<T> A, T b) {
            Tensor<T> C(A.shape(), A.device(), false);
            cupp::Library::error_check(cupp::cu_sub_with_rconst(A.View(), &b, C.View(), A.total_size(), A.dtype()));
            return C;
        }

    };

    template<typename T>
    class Sub_with_lconst_op : public Tensor_op<T> {
    public:
        Sub_with_lconst_op() = default;

        Tensor<T> forward(T& a, Tensor<T>& B) {
            Tensor<T> C = sub_with_lconst(a, B);

            B.used_count()++;

            C.required_grad() = true;
            C.tensor_from().push_back(B);
            C.op() = this;
            return C;
        }

        std::vector<Tensor<T>> backward(std::vector<Tensor<T>>& tensor_from, Tensor<T>& grad_in) override {
            std::vector<Tensor<T>> grad_out;
            grad_out.push_back(Opposite_op<T>::opposite(grad_in));
            return grad_out;
        }

        Tensor<T> static sub_with_lconst(T a, Tensor<T> B) {
            Tensor<T> C(B.shape(), B.device(), false);
            cupp::Library::error_check(cupp::cu_sub_with_lconst(&a, B.View(), C.View(), B.total_size(), B.dtype()));
            return C;
        }
    };

    template<typename T>
    class Transpose_op : public Tensor_op<T> {
    public:
        Transpose_op() = default;

        Tensor<T> forward(Tensor<T>& A) {
            Tensor<T> B = transpose(A);

            A.used_count()++;

            B.required_grad() = true;
            B.tensor_from().push_back(A);
            B.op() = this;
            return B;
        }

        std::vector<Tensor<T>> backward(std::vector<Tensor<T>>& tensor_from, Tensor<T>& grad_in) override {
            std::vector<Tensor<T>> grad_out;
            grad_out.push_back(transpose(grad_in));
            return grad_out;
        }

        Tensor<T> static transpose(Tensor<T> A) {
#ifdef _DEBUG
            if (A.shape().size() < 2)
            {
                std::cout << "Tensor_operation_error:Transpose_op-> The shape of Tensor: " << A.shape() << " should be at least 2D" << "\n";
                throw std::invalid_argument("Tensor_operation_error:Transpose_op-> Tensor shape must should be at least 2D");
            }
#endif
            std::vector<long long> B_shape = A.shape();
            B_shape[B_shape.size() - 2] = A.shape()[A.shape().size() - 1];
            B_shape[B_shape.size() - 1] = A.shape()[A.shape().size() - 2];
            Tensor<T> B(B_shape, A.device(), false);
            cupp::Library::error_check(cupp::cu_transpose(A.View(), B.View(), A.shape()[A.shape().size() - 2], A.shape()[A.shape().size() - 1], A.matmul_batch(), A.dtype()));
            return B;

        }

    };

    //transpose_with_dim
    template<typename T>
    class Transpose_with_dim_op : public Tensor_op<T> {
    private:
        long long dim1;
        long long dim2;
    public:
        Transpose_with_dim_op() = default;

        Tensor<T> forward(Tensor<T>& A, long long& dim1, long long& dim2) {
            Tensor<T> B = transpose_with_dim(A, dim1, dim2);
            this->dim1 = dim1;
            this->dim2 = dim2;

            A.used_count()++;
            B.required_grad() = true;
            B.tensor_from().push_back(A);
            B.op() = this;
            return B;
        }

        std::vector<Tensor<T>> backward(std::vector<Tensor<T>>& tensor_from, Tensor<T>& grad_in) override {
            std::vector<Tensor<T>> grad_out;
            grad_out.push_back(transpose_with_dim(grad_in, dim1, dim2));
            return grad_out;
        }

        Tensor<T> static transpose_with_dim(Tensor<T> A, long long dim1, long long dim2) {
#ifdef _DEBUG
            if (dim1 == dim2) {
                std::cout << "Tensor_operation_error:Transpose_with_dim_op-> dim1 and dim2 can't be the same," << " dim1:" << dim1 << " dim2:" << dim2 << "\n";
                throw std::invalid_argument("Tensor_operation_error:Transpose_with_dim_op-> dim1 and dim2 can't be the same");
            }
            if (A.shape().size() < 2) {
                std::cout << "Tensor_operation_error:Transpose_with_dim_op-> The shape of Tensor: " << A.shape() << " should be at least 2D" << "\n";
                throw std::invalid_argument("Tensor_operation_error:Transpose_with_dim_op-> Tensor shape must should be at least 2D");
            }
            if (dim1 >= A.shape().size() || dim2 >= A.shape().size()) {
                std::cout << "Tensor_operation_error:Transpose_with_dim_op-> dim_out of range, dim1:" << dim1 << " dim2:" << dim2 << " A_shape:" << A.shape() << "\n";
                throw std::invalid_argument("Tensor_operation_error:Transpose_with_dim_op-> dim_out of range");
            }
            if (dim1 < 0 || dim2 < 0) {
                std::cout << "Tensor_operation_error:Transpose_with_dim_op-> dim_must should be non-negative, dim1:" << dim1 << " dim2:" << dim2 << "\n";
                throw std::invalid_argument("Tensor_operation_error:Transpose_with_dim_op-> dim_must should be non-negative");
            }
#endif
            std::vector<long long> B_shape(A.shape());
            B_shape[dim1] = A.shape()[dim2];
            B_shape[dim2] = A.shape()[dim1];
            Tensor<T> B(B_shape, A.device(), false);

            const long long& max_dim = dim1 > dim2 ? dim1 : dim2;
            const long long& min_dim = dim1 < dim2 ? dim1 : dim2;
            long long between_batch = 1;
            for (long long i = min_dim + 1; i < max_dim; i++) {
                between_batch *= A.shape()[i];
            }

            long long behind_batch = 1;
            for (long long i = max_dim + 1; i < A.shape().size(); i++) {
                behind_batch *= A.shape()[i];
            }
            cupp::Library::error_check(cupp::cu_transpose_with_dim(A.View(), B.View(), A.total_size(), A.shape()[min_dim], between_batch, A.shape()[max_dim], behind_batch, A.dtype()));
            return B;
        }

    };

    class cublas_IntPtr {
    private:
        void* cublas_handle = nullptr;
    public:
        cublas_IntPtr() {
            cupp::Library::error_check(cupp::cublas_handle_create(cublas_handle));
        }

        void* get_handle() {
            return cublas_handle;
        }


        ~cublas_IntPtr() {
            if (cublas_handle != nullptr) {
                cupp::Library::error_check(cupp::cublas_handle_destroy(cublas_handle));
            }
        }

        cublas_IntPtr(const cublas_IntPtr&) = delete;

        cublas_IntPtr& operator=(const cublas_IntPtr&) = delete;

        cublas_IntPtr(cublas_IntPtr&&) = delete;

        cublas_IntPtr& operator=(cublas_IntPtr&&) = delete;

    };

    inline void* cublas_handle() {
        static cublas_IntPtr cublas_handle;
        return cublas_handle.get_handle();
    }

    template<typename T>
    class Matmul_op : public Tensor_op<T> {
    public:

        Matmul_op() = default;


        Tensor<T> forward(Tensor<T>& A, Tensor<T>& B) {
            Tensor<T> C = matmul(A, B);

            A.used_count()++;
            B.used_count()++;

            C.required_grad() = true;
            C.tensor_from().push_back(A);
            C.tensor_from().push_back(B);
            C.op() = this;
            return C;
        }

        std::vector<Tensor<T>> backward(std::vector<Tensor<T>>& tensor_from, Tensor<T>& grad_in) override {
            std::vector<Tensor<T>> grad_out;
            grad_out.push_back(batch_down(matmul(grad_in, Transpose_op<T>::transpose(tensor_from[1])), tensor_from[0]));
            grad_out.push_back(batch_down(matmul(Transpose_op<T>::transpose(tensor_from[0]), grad_in), tensor_from[1]));
            return grad_out;
        }

        Tensor<T> static matmul(Tensor<T> A, Tensor<T> B) {
#ifdef _DEBUG
            if (A.shape().size() < 2 || B.shape().size() < 2)
            {
                std::cout << "Tensor_operation_error:Matmul_op-> A_shape:" << A.shape() << " B_shape:" << B.shape() << " should be at least 2D" << "\n";
                throw std::invalid_argument("Tensor_operation_error:Matmul_op-> A,B shape should be at least 2D");
            }
            if (A.shape()[A.shape().size() - 1] != B.shape()[B.shape().size() - 2])
            {
                std::cout << "Tensor_operation_error:Matmul_op-> A_shape:" << A.shape() << " B_shape:" << B.shape() << " last dim of A should be equal to second last dim of B" << "\n";
                throw std::invalid_argument("Tensor_operation_error:Matmul_op-> A,B shape mismatch for matmul");
            }
#endif
            std::vector<long long> C_shape = Broadcast_rule(A.shape(), B.shape(), 2, "Matmul_op->");
            C_shape[C_shape.size() - 2] = A.shape()[A.shape().size() - 2];
            C_shape[C_shape.size() - 1] = B.shape()[B.shape().size() - 1];
            Tensor<T> C(C_shape, A.device(), false);

            long m = A.shape()[A.shape().size() - 2];
            long n = B.shape()[B.shape().size() - 1];
            long k = A.shape()[A.shape().size() - 1];
            cupp::Library::error_check(cupp::cublas_matmul(cublas_handle(), A.View(), B.View(), C.View(), A.matmul_batch(), B.matmul_batch(), m, n, k, A.dtype()));
            return C;
        }


    };

    //Hadamard
    template<typename T>
    class Hadamard_op : public Tensor_op<T> {
    public:
        Hadamard_op() = default;

        Tensor<T> forward(Tensor<T>& A, Tensor<T>& B) {
            Tensor<T> C = hadamard(A, B);

            A.used_count()++;
            B.used_count()++;
            C.required_grad() = true;
            C.tensor_from().push_back(A);
            C.tensor_from().push_back(B);
            C.op() = this;
            return C;
        }

        std::vector<Tensor<T>> backward(std::vector<Tensor<T>>& tensor_from, Tensor<T>& grad_in) override {
            std::vector<Tensor<T>> grad_out;
            grad_out.push_back(batch_down(hadamard(grad_in, tensor_from[1]), tensor_from[0]));
            grad_out.push_back(batch_down(hadamard(tensor_from[0], grad_in), tensor_from[1]));
            return grad_out;
        }

        Tensor<T> static hadamard(Tensor<T> A, Tensor<T> B) {
            std::vector<long long> C_shape = Broadcast_rule(A.shape(), B.shape(), 0, "Hadamard_op->");
            Tensor<T> C(C_shape, A.device(), false);
            cupp::Library::error_check(cupp::cu_hadamard(A.View(), B.View(), C.View(), A.total_size(), B.total_size(), A.dtype()));
            return C;
        }

    };

    //Hadamard_with_const
    template<typename T>
    class Hadamard_with_const_op : public Tensor_op<T> {
    private:
        T b;
    public:
        Hadamard_with_const_op() = default;

        Tensor<T> forward(Tensor<T>& A, T& b) {
            Tensor<T> C = hadamard_with_const(A, b);

            this->b = b;
            A.used_count()++;

            C.required_grad() = true;
            C.tensor_from().push_back(A);
            C.op() = this;
            return C;
        }

        std::vector<Tensor<T>> backward(std::vector<Tensor<T>>& tensor_from, Tensor<T>& grad_in) override {
            std::vector<Tensor<T>> grad_out;
            grad_out.push_back(hadamard_with_const(grad_in, this->b));
            return grad_out;
        }

        Tensor<T> static hadamard_with_const(Tensor<T> A, T b) {
            Tensor<T> C(A.shape(), A.device(), false);
            cupp::Library::error_check(cupp::cu_hadamard_with_const(A.View(), &b, C.View(), A.total_size(), A.dtype()));
            return C;
        }


    };

    //Division
    template<typename T>
    class Division_op : public Tensor_op<T> {
    public:
        Division_op() = default;

        Tensor<T> forward(Tensor<T>& A, Tensor<T>& B) {
            Tensor<T> C = division(A, B);

            A.used_count()++;
            B.used_count()++;

            C.required_grad() = true;
            C.tensor_from().push_back(A);
            C.tensor_from().push_back(B);
            C.op() = this;
            return C;
        }

        std::vector<Tensor<T>> backward(std::vector<Tensor<T>>& tensor_from, Tensor<T>& grad_in) override {
            std::vector<Tensor<T>> grad_out;
            grad_out.push_back(batch_down(division(grad_in, tensor_from[1]), tensor_from[0]));
            grad_out.push_back(batch_down(Hadamard_op<T>::hadamard(grad_in, division(Opposite_op<T>::opposite(tensor_from[0]), Hadamard_op<T>::hadamard(tensor_from[1], tensor_from[1]))), tensor_from[1]));
            return grad_out;
        }

        Tensor<T> static division(Tensor<T> A, Tensor<T> B) {
            std::vector<long long> C_shape = Broadcast_rule(A.shape(), B.shape(), 0, "Division_op->");
            Tensor<T> C(C_shape, A.device(), false);
            cupp::Library::error_check(cupp::cu_division(A.View(), B.View(), C.View(), A.total_size(), B.total_size(), A.dtype()));
            return C;
        }
    };

    //Division_with_Rconst
    template<typename T>
    class Division_with_rconst_op : public Tensor_op<T> {
    private:
        T b;
    public:
        Division_with_rconst_op() = default;

        Tensor<T> forward(Tensor<T>& A, T& b) {
            Tensor<T> C = division_with_rconst(A, b);
            this->b = b;
            A.used_count()++;

            C.required_grad() = true;
            C.tensor_from().push_back(A);
            C.op() = this;
            return C;
        }

        std::vector<Tensor<T>> backward(std::vector<Tensor<T>>& tensor_from, Tensor<T>& grad_in) override {
            std::vector<Tensor<T>> grad_out;
            grad_out.push_back(division_with_rconst(grad_in, this->b));
            return grad_out;
        }


        Tensor<T> static division_with_rconst(Tensor<T> A, T b) {
            Tensor<T> C(A.shape(), A.device(), false);
            cupp::Library::error_check(cupp::cu_division_with_rconst(A.View(), &b, C.View(), A.total_size(), A.dtype()));
            return C;
        }

    };


    //Division_with_Lconst
    template<typename T>
    class Division_with_lconst_op : public Tensor_op<T> {
    private:
        T a;
    public:
        Division_with_lconst_op() = default;

        Tensor<T> forward(T& a, Tensor<T>& B) {
            Tensor<T> C = division_with_lconst(a, B);
            this->a = a;
            B.used_count()++;

            C.required_grad() = true;
            C.tensor_from().push_back(B);
            C.op() = this;
            return C;
        }

        std::vector<Tensor<T>> backward(std::vector<Tensor<T>>& tensor_from, Tensor<T>& grad_in) override {
            std::vector<Tensor<T>> grad_out;
            grad_out.push_back(Hadamard_op<T>::hadamard(Opposite_op<T>::opposite(grad_in), division_with_lconst(a, Hadamard_op<T>::hadamard(tensor_from[0], tensor_from[0]))));
            return grad_out;
        }

        Tensor<T> static division_with_lconst(T a, Tensor<T> B) {
            Tensor<T> C(B.shape(), B.device(), false);
            cupp::Library::error_check(cupp::cu_division_with_lconst(&a, B.View(), C.View(), B.total_size(), B.dtype()));
            return C;
        }

    };

    //Pow
    template<typename T>
    class Pow_op : public Tensor_op<T> {
    private:
        int pow_number;
    public:
        Pow_op() = default;

        Tensor<T> forward(Tensor<T>& A, int& pow_number) {
            Tensor<T> C = pow(A, pow_number);

            this->pow_number = pow_number;
            A.used_count()++;
            C.required_grad() = true;
            C.tensor_from().push_back(A);
            C.op() = this;
            return C;
        }

        std::vector<Tensor<T>> backward(std::vector<Tensor<T>>& tensor_from, Tensor<T>& grad_in) override {
            std::vector<Tensor<T>> grad_out;
            grad_out.push_back(Hadamard_op<T>::hadamard(grad_in, Hadamard_with_const_op<T>::hadamard_with_const(pow(tensor_from[0], this->pow_number - 1), (T)pow_number)));
            return grad_out;
        }

        Tensor<T> static pow(Tensor<T> A, int pow_number) {
#ifdef _DEBUG
            if (pow_number < 0) {
                std::cout << "Tensor_operation_error:Pow_op-> pow_number must be a positive number." << "\n";
                throw std::invalid_argument("Tensor_operation_error:Pow_op-> pow_number must be a positive number.");
            }
#endif
            Tensor<T> B = A.Copy();
            if (pow == 0) {
                B.One();
                return B;
            }
            for (int i = 0; i < pow_number - 1; i++) {
                B = Hadamard_op<T>::hadamard(A, B);
            }
            return B;

        }

    };

    //sqrt
    template<typename T>
    class Sqrt_op : Tensor_op<T> {
    public:
        Sqrt_op() = default;

        Tensor<T> forward(Tensor<T>& A) {
            Tensor<T> B = sqrt(A);

            A.used_count()++;
            B.required_grad() = true;
            B.tensor_from().push_back(A);
            B.op() = this;
            return B;
        }

        std::vector<Tensor<T>> backward(std::vector<Tensor<T>>& tensor_from, Tensor<T>& grad_in) override {
            std::vector<Tensor<T>> grad_out;
            grad_out.push_back(Division_op<T>::division(Division_with_rconst_op<T>::division_with_rconst(grad_in, (T)2), sqrt(tensor_from[0])));
            return grad_out;
        }

        Tensor<T> static sqrt(Tensor<T> A) {
            Tensor<T> B(A.shape(), A.device(), false);
            cupp::Library::error_check(cupp::cu_sqrt(A.View(), B.View(), A.total_size(), A.dtype()));
            return B;
        }

    };

    //exp
    template<typename T>
    class Exp_op : public Tensor_op<T> {
    public:
        Exp_op() = default;

        Tensor<T> forward(Tensor<T>& A) {
            Tensor<T> B = exp(A);
            A.used_count()++;
            B.required_grad() = true;
            B.tensor_from().push_back(A);
            B.op() = this;
            return B;
        }

        std::vector<Tensor<T>> backward(std::vector<Tensor<T>>& tensor_from, Tensor<T>& grad_in) override {
            std::vector<Tensor<T>> grad_out;
            grad_out.push_back(Hadamard_op<T>::hadamard(grad_in, exp(tensor_from[0])));
            return grad_out;
        }

        Tensor<T> static exp(Tensor<T> A) {
            Tensor<T> B(A.shape(), A.device(), false);
            cupp::Library::error_check(cupp::cu_exp(A.View(), B.View(), A.total_size(), A.dtype()));
            return B;
        }

    };

    //ln
    template<typename T>
    class Ln_op : public Tensor_op<T> {
    public:
        Ln_op() = default;

        Tensor<T> forward(Tensor<T>& A) {
            Tensor<T> B = ln(A);

            A.used_count()++;
            B.required_grad() = true;
            B.tensor_from().push_back(A);
            B.op() = this;
            return B;
        }

        std::vector<Tensor<T>> backward(std::vector<Tensor<T>>& tensor_from, Tensor<T>& grad_in) override {
            std::vector<Tensor<T>> grad_out;
            grad_out.push_back(Division_op<T>::division(grad_in, tensor_from[0]));
            return grad_out;
        }

        Tensor<T> static ln(Tensor<T> A) {
            Tensor<T> B(A.shape(), A.device(), false);
            cupp::Library::error_check(cupp::cu_ln(A.View(), B.View(), A.total_size(), A.dtype()));
            return B;
        }

    };

    //cos
    template<typename T>
    class Cos_op : public Tensor_op<T> {
    public:
        Cos_op() = default;

        Tensor<T> forward(Tensor<T>& A) {
            Tensor<T> B = cos(A);
            A.used_count()++;

            B.required_grad() = true;
            B.tensor_from().push_back(A);
            B.op() = this;
            return B;
        }

        std::vector<Tensor<T>> backward(std::vector<Tensor<T>>& tensor_from, Tensor<T>& grad_in) override;

        Tensor<T> static cos(Tensor<T> A) {
            Tensor<T> B(A.shape(), A.device(), false);
            cupp::Library::error_check(cupp::cu_cos(A.View(), B.View(), A.total_size(), A.dtype()));
            return B;
        }

    };

    //sin
    template<typename T>
    class Sin_op : public Tensor_op<T> {
    public:
        Sin_op() = default;

        Tensor<T> forward(Tensor<T>& A) {
            Tensor<T> B = sin(A);
            A.used_count()++;

            B.required_grad() = true;
            B.tensor_from().push_back(A);
            B.op() = this;
            return B;
        }

        std::vector<Tensor<T>> backward(std::vector<Tensor<T>>& tensor_from, Tensor<T>& grad_in) override;

        Tensor<T> static sin(Tensor<T> A) {
            Tensor<T> B(A.shape(), A.device(), false);
            cupp::Library::error_check(cupp::cu_sin(A.View(), B.View(), A.total_size(), A.dtype()));
            return B;
        }
    };

    //reshape
    template<typename T>
    class Reshape_op : public Tensor_op<T> {
    public:
        Reshape_op() = default;

        Tensor<T> forward(Tensor<T>& A, std::vector<long long>& reshape_shape) {
            Tensor<T> B = reshape(A, reshape_shape);
            A.used_count()++;

            B.required_grad() = true;
            B.tensor_from().push_back(A);
            B.op() = this;
            return B;
        }

        std::vector<Tensor<T>> backward(std::vector<Tensor<T>>& tensor_from, Tensor<T>& grad_in) override {
            std::vector<Tensor<T>> grad_out;
            grad_out.push_back(reshape(grad_in, tensor_from[0].shape()));
            return grad_out;
        }

        Tensor<T> static reshape(Tensor<T> A, std::vector<long long> reshape_shape) {
            long long reshape_total_size = 1;
            for (long long dim : reshape_shape)
            {
#ifdef _DEBUG
                if (dim <= 0) {
                    std::cout << "Tensor_operation_error:Reshape_op-> error_shape_input:" << reshape_shape << "\n";
                    throw std::invalid_argument("Tensor_operation_error:Reshape_op-> error_shape_input");
                }
#endif
                reshape_total_size *= dim;
            }
#ifdef _DEBUG
            if (reshape_total_size != A.total_size()) {
                std::cout << "Tensor_operation_error:Reshape_op-> original_shape_total_size:" << A.total_size() << " reshape_shape_total_size" << reshape_total_size << " must equal" << "\n";
                throw std::invalid_argument("Tensor_operation_error:Reshape_op-> reshape_shape_total_size must equal to original_shape_total_size");
            }
#endif
            Tensor<T> B(reshape_shape, A.device(), false);
            cupp::Library::error_check(cupp::cu_Copy(A.View(), B.View(), A.total_size(), A.dtype()));
            return B;
        }

    };

    //sum_in_dim
    template<typename T>
    class Sum_in_dim_op : public Tensor_op<T> {
    private:
        long long dim;
        long long dim_in_shape;
    public:
        Sum_in_dim_op() = default;

        Tensor<T> forward(Tensor<T>& A, long long& dim) {
            Tensor<T> B = sum_in_dim(A, dim);

            this->dim = dim;
            this->dim_in_shape = A.shape()[dim];

            A.used_count()++;

            B.required_grad() = true;
            B.tensor_from().push_back(A);
            B.op() = this;
            return B;
        }

        std::vector<Tensor<T>> backward(std::vector<Tensor<T>>& tensor_from, Tensor<T>& grad_in) override;

        Tensor<T> static sum_in_dim(Tensor<T> A, long long dim) {
#ifdef _DEBUG
            if (dim < 0) {
                std::cout << "Tensor_operation_error:Sum_in_dim_op-> dim_index should be non-negative" << "\n";
                throw std::invalid_argument("Tensor_operation_error:Sum_in_dim_op-> dim_index should be non-negative");
            }
            if (dim >= A.shape().size()) {
                std::cout << "Tensor_operation_error:Sum_in_dim_op-> " << "dim_in_sum_in_dim" << dim << "out of range" << "\n";
                throw std::invalid_argument("Tensor_operation_error:Sum_in_dim_op-> dim out of range");
            }
#endif
            std::vector<long long> B_shape(A.shape().size() - 1);

            for (long long j = 0; j < dim; j++) {
                B_shape[j] = A.shape()[j];
            }

            for (long long i = dim + 1; i < A.shape().size(); i++) {
                B_shape[i - 1] = A.shape()[i];
            }


            long long back_total_size = 1;
            for (long long i = dim + 1; i < A.shape().size(); i++) {
                back_total_size *= A.shape()[i];
            }

            Tensor<T> B(B_shape, A.device(), false);
            cupp::Library::error_check(cupp::cu_sum_in_dim(A.View(), B.View(), B.total_size(), back_total_size, A.shape()[dim], A.dtype()));
            return B;
        }

    };

    //broadcast_in_dim_op
    template<typename T>
    class Broadcast_in_dim_op : public Tensor_op<T> {
    private:
        long long dim;
    public:
        Broadcast_in_dim_op() = default;

        Tensor<T> forward(Tensor<T>& A, long long& up_dim, long long& up_dim_in_shape) {
            Tensor<T> B = broadcast_in_dim(A, up_dim, up_dim_in_shape);
            this->dim = up_dim;

            A.used_count()++;

            B.required_grad() = true;
            B.tensor_from().push_back(A);
            B.op() = this;
            return B;
        }

        std::vector<Tensor<T>> backward(std::vector<Tensor<T>>& tensor_from, Tensor<T>& grad_in) override;

        Tensor<T> static broadcast_in_dim(Tensor<T> A, long long up_dim, long long up_dim_in_shape) {
#ifdef _DEBUG
            if (up_dim_in_shape < 0) {
                std::cout << "Tensor_operation_error:Broadcast_in_dim_op-> dim_index should be non-negative" << "\n";
                throw std::invalid_argument("Tensor_operation_error:Broadcast_in_dim_op-> dim_index should be non-negative");
            }

            if (up_dim > A.shape().size() || up_dim < 0) {
                std::cout << "Tensor_operation_error:Broadcast_in_dim_op-> " << "dim_in_sum_in_dim" << up_dim << "out of range" << "\n";
                throw std::invalid_argument("Tensor_operation_error:Broadcast_in_dim_op-> dim out of range");
            }
#endif
            std::vector <long long> B_shape(A.shape().size() + 1);
            B_shape[up_dim] = up_dim_in_shape;
            for (long long i = 0; i < up_dim; i++) {
                B_shape[i] = A.shape()[i];
            }

            for (long long i = up_dim; i < A.shape().size(); i++) {
                B_shape[i + 1] = A.shape()[i];
            }

            long long back_total_size = 1;
            for (long long i = up_dim + 1; i < A.shape().size(); i++) {
                back_total_size *= A.shape()[i];
            }

            Tensor<T> B(B_shape, A.device(), false);

            cupp::Library::error_check(cupp::cu_dim_up(A.View(), B.View(), B.total_size(), back_total_size, up_dim_in_shape, A.dtype()));
            return B;
        }
    };

    template<typename T>
    inline static T cu_total_sum(Tensor<T> A) {
        static cublas_IntPtr cublas_handle;
        T a;
        cupp::Library::error_check(cupp::cu_total_sum(cublas_handle.get_handle(), A.View(), &a, A.total_size(), A.dtype()));
        return a;
    }


    //ref_operation
    template<typename T>
    inline static void ref_add(Tensor<T> A, Tensor<T> B) {
        Broadcast_rule(A.shape(), B.shape(), 0, "ref_add_op->");
        cupp::Library::error_check(cupp::cu_ref_add(A.View(), B.View(), A.total_size(), B.total_size(), A.dtype()));
    }

    template<typename T>
    inline static void ref_sub(Tensor<T> A, Tensor<T> B) {
        Broadcast_rule(A.shape(), B.shape(), 0, "ref_sub_op->");
        cupp::Library::error_check(cupp::cu_ref_sub(A.View(), B.View(), A.total_size(), B.total_size(), A.dtype()));
    }

    template<typename T>
    inline static void ref_hadamard(Tensor<T> A, Tensor<T> B) {
        Broadcast_rule(A.shape(), B.shape(), 0, "ref_hadamard_op->");
        cupp::Library::error_check(cupp::cu_ref_hadamard(A.View(), B.View(), A.total_size(), B.total_size(), A.dtype()));
    }

    template<typename T>
    inline static void ref_division(Tensor<T> A, Tensor<T> B) {
        Broadcast_rule(A.shape(), B.shape(), 0, "ref_division_op->");
        cupp::Library::error_check(cupp::cu_ref_division(A.View(), B.View(), A.total_size(), B.total_size(), A.dtype()));
    }

    class cudnn_IntPtr {
    private:
        void* cudnn_handle = nullptr;
    public:
        cudnn_IntPtr() {
            cupp::Library::error_check(cupp::cudnn_handle_create(cudnn_handle));
        }

        void* get_handle() {
            return cudnn_handle;
        }


        ~cudnn_IntPtr() {
            if (cudnn_handle != nullptr) {
                cupp::Library::error_check(cupp::cudnn_handle_destroy(cudnn_handle));
            }
        }

        cudnn_IntPtr(const cudnn_IntPtr&) = delete;

        cudnn_IntPtr& operator=(const cudnn_IntPtr&) = delete;

        cudnn_IntPtr(cudnn_IntPtr&&) = delete;

        cudnn_IntPtr& operator=(cudnn_IntPtr&&) = delete;

    };

    inline void* cudnn_handle() {
        static cudnn_IntPtr _cudnn;
        return _cudnn.get_handle();
    }

    //cudnn_accelerate
    template<typename T>
    class Conv2D_op : public Tensor_op<T> {
    private:
        std::vector<long long> stride;
        std::vector<long long> padding;
    public:
        Conv2D_op() = default;

        Tensor<T> forward(Tensor<T>& A, Tensor<T>& kernel, std::vector<long long> stride, std::vector<long long> padding) {
            Tensor<T> C = conv2d_forward(A, kernel, stride, padding);

            this->stride = stride;
            this->padding = padding;
            A.used_count()++;
            kernel.used_count()++;

            C.required_grad() = true;
            C.tensor_from().push_back(A);
            C.tensor_from().push_back(kernel);
            C.op() = this;
            return C;
        }

        std::vector<Tensor<T>> backward(std::vector<Tensor<T>>& tensor_from, Tensor<T>& grad_in) override {
            std::vector<Tensor<T>> grad_out;
            grad_out.push_back(conv2d_backward_data(grad_in, tensor_from[1], tensor_from[0].shape(), stride, padding));
            grad_out.push_back(conv2d_backward_kernel(grad_in, tensor_from[0], tensor_from[1].shape(), stride, padding));
            return grad_out;
        }


        Tensor<T> static conv2d_forward(Tensor<T> A, Tensor<T> kernel, std::vector<long long> stride, std::vector<long long> padding) {
#ifdef _DEBUG
            std::vector<long long> A_shape;
            if (A.shape().size() == 3) {
                A_shape = std::vector<long long>(4);
                A_shape[0] = 1;
                for (int i = 0; i < 3; i++) {
                    A_shape[i + 1] = A.shape()[i];
                }
            }
            else if (A.shape().size() == 4)
                A_shape = A.shape();
            else {
                std::cout << "Tensor_operation_error:convolution2D_op-> shape_error:A_shape->" << A.shape() << "\n";
                throw std::invalid_argument("Tensor_operation_error:convolution2D_op-> Tensor shape must be NCHW");
            }

            if (kernel.shape().size() != 4) {
                std::cout << "Tensor_operation_error:convolution2D_op-> kernel_shape->" << kernel.shape() << "\n";
                throw std::invalid_argument("Tensor_operation_error:convolution2D_op-> Tensor shape must be NCHW ,kernel_shape must be out_channel,in_channel,kernel_x,kernel_y");
            }
            if (stride.size() != 2 || padding.size() != 2) {
                std::cout << "Tensor_operation_error:convolution2D_op-> Invalid parameter:stride->" << stride << ",padding->" << padding << "\n";
                throw std::invalid_argument("Tensor_operation_error:convolution2D_op-> stride padding size must be equal to two.");
            }
            if (stride[0] < 0 || stride[1] < 0 || padding[0] < 0 || padding[1] < 0) {
                std::cout << "Tensor_operation_error:convolution2D_op-> Invalid parameter stride_x->" << stride[0] << ",stride_y" << stride[1] << ",padding_x" << padding[0] << ",padding_y" << padding[1] << "\n";
                throw std::invalid_argument("Tensor_operation_error:convolution2D_op-> Tensor shape must be NCHW ,kernel_shape must be out_channel,in_channel,kernel_x,kernel_y");
            }
            if (A_shape[1] <= 0 || kernel.shape()[0] <= 0) {
                std::cout << "Tensor_operation_error:convolution2D_op-> in_channel:" << A_shape[1] << ",out_channel:" << kernel.shape()[0] << "\n";
                throw std::invalid_argument("Tensor_operation_error:convolution2D_op-> in_channel and out_channel must be positive integers");
            }
            if (A_shape[1] != kernel.shape()[1]) {
                std::cout << "Tensor_operation_error:convolution2D_op->input in_channel:" << A_shape[1] << " , kernel in_channel:" << kernel.shape()[1] << " must be equal" << "\n";
                throw std::invalid_argument("Tensor_operation_error:convolution2D_op-> input in_channel and kernel in_channel must be equal");
            }
#endif

            long long out_x = (A_shape[2] + 2 * padding[0] - kernel.shape()[2]) / stride[0] + 1;
            long long out_y = (A_shape[3] + 2 * padding[1] - kernel.shape()[3]) / stride[1] + 1;
            std::vector<long long> C_shape = A_shape;
            C_shape[1] = kernel.shape()[0];
            C_shape[2] = out_x;
            C_shape[3] = out_y;

            Tensor<T> C(C_shape, A.device(), false);
            cupp::Library::error_check(cupp::cudnn_conv2d_forward(cudnn_handle(), A.View(), kernel.View(), C.View(), A.conv2d_batch(), A_shape[1], kernel.shape()[0], A_shape[2], A_shape[3], kernel.shape()[2], kernel.shape()[3], stride[0], stride[1], padding[0], padding[1], A.dtype()));
            return C;
        }

        Tensor<T> static conv2d_backward_data(Tensor<T> in_grad, Tensor<T> kernel, std::vector<long long> out_grad_shape, std::vector<long long> stride, std::vector<long long> padding) {
            Tensor<T> C(out_grad_shape, in_grad.device(), false);
            cupp::Library::error_check(cupp::cudnn_conv2d_backward_data(cudnn_handle(), in_grad.View(), kernel.View(), C.View(), in_grad.conv2d_batch(), out_grad_shape[1], in_grad.shape()[1], in_grad.shape()[2], in_grad.shape()[3], out_grad_shape[2], out_grad_shape[3], kernel.shape()[2], kernel.shape()[3], stride[0], stride[1], padding[0], padding[1], in_grad.dtype()));
            return C;
        }

        Tensor<T> static conv2d_backward_kernel(Tensor<T> in_grad, Tensor<T> input_tensor, std::vector<long long> out_grad_shape, std::vector<long long> stride, std::vector<long long> padding) {
            Tensor<T> C(out_grad_shape, in_grad.device(), false);
            cupp::Library::error_check(cupp::cudnn_conv2d_backward_kernel(cudnn_handle(), in_grad.View(), input_tensor.View(), C.View(), in_grad.conv2d_batch(), input_tensor.shape()[1], in_grad.shape()[1], in_grad.shape()[2], in_grad.shape()[3], input_tensor.shape()[2], input_tensor.shape()[3], out_grad_shape[2], out_grad_shape[3], stride[0], stride[1], padding[0], padding[1], in_grad.dtype()));
            return C;
        }
    };

    template<typename T>
    class Batch_cat_op :public Tensor_op<T> {
    private:
        std::vector<long long> sample_shape;
        long long sample_total_size = 0;
    public:
        Batch_cat_op() = default;

        Tensor<T> forward(std::vector<Tensor<T>>& tensor_list) {
            Tensor<T> C = batch_cat_forward(tensor_list);
            
           this->sample_shape = tensor_list[0].shape();
           this->sample_total_size = tensor_list[0].total_size();
            for (long long i = 0; i < tensor_list.size(); i++) {
                tensor_list[i].used_count()++;
                C.tensor_from().push_back(tensor_list[i]);
            }
            C.required_grad() = true;
            C.op() = this;
            return C;
        }

        std::vector<Tensor<T>> backward(std::vector<Tensor<T>>& tensor_from, Tensor<T>& grad_in) override {
            std::vector<Tensor<T>> grad_out;
            long long cat_batch = grad_in.total_size() / this->sample_total_size;
            long long ptr_offsets = 0;
            for (long long i = 0; i < cat_batch; i++) {
                Tensor<T> C(this->sample_shape, grad_in.device(), false);
                cupp::Library::error_check(cupp::cu_Copy(grad_in.View() + ptr_offsets, C.View(), this->sample_total_size, grad_in.dtype()));
                ptr_offsets += this->sample_total_size;
                grad_out.push_back(C);
            }
            return grad_out;
        }

        Tensor<T> static batch_cat_forward(std::vector<Tensor<T>> tensor_list) {
            std::vector<long long> C_shape(tensor_list[0].shape().size() + 1);
            C_shape[0] = tensor_list.size();
            for (long long i = 0; i < tensor_list[0].shape().size(); i++) {
                C_shape[i + 1] = tensor_list[0].shape()[i];
            }
            Tensor<T> C(C_shape, tensor_list[0].device(), false);
            long long ptr_offsets = 0;
            for (long long i = 0; i < tensor_list.size(); i++) {
#ifdef _DEBUG
                if (tensor_list[0].shape() != tensor_list[i].shape()) {
                    std::cout << "Tensor_operation_error:Batch_cat_op->input Tensor list:" << i << "->" << tensor_list[i].shape() << "different from the previous shape->" << tensor_list[0].shape() << "\n";
                    throw std::invalid_argument("Tensor_operation_error:Batch_cat_op->irregular shapes in the tensor list.");
                }
#endif
                cupp::Library::error_check(cupp::cu_Copy(tensor_list[i].View(), C.View() + ptr_offsets, tensor_list[i].total_size(), tensor_list[i].dtype()));
                ptr_offsets += tensor_list[i].total_size();
            }
            return C;
        }

    };

    template<typename T>
    class PPO_clip_op :public Tensor_op<T> {
    private:
        Tensor<T> gd_mask;
    public:
        PPO_clip_op() = default;

        Tensor<T> forward(Tensor<T>& A,T down_limit,T up_limit) {
            if (gd_mask.initialize() == false)
                gd_mask = Tensor<T>(A.shape(),A.device(),false);
            Tensor<T> B = cu_ppo_clip(A,gd_mask,down_limit,up_limit);

            A.used_count()++;
            B.tensor_from().push_back(A);
            B.required_grad() = true;
            B.op() = this;
            return B;
        }

        std::vector<Tensor<T>> backward(std::vector<Tensor<T>>& tensor_from, Tensor<T>& grad_in) override {
            std::vector<Tensor<T>> grad_out;
            grad_out.push_back(Hadamard_op<T>::hadamard(grad_in,this->gd_mask));
            return grad_out;
        }


        Tensor<T> static cu_ppo_clip(Tensor<T> A,Tensor<T> gd_mask,T down_limit,T up_limit) {
            Tensor<T> B(A.shape(),A.device(),false);
            cupp::Library::error_check(cupp::cu_ppo_clip(A.View(),B.View(),gd_mask.View(),&down_limit,&up_limit,A.total_size(),A.dtype()));
            return B;
        }
    };

    template<typename T>
    class Min_op :public Tensor_op<T> {
    private:
        Tensor<T> A_gdmask;
        Tensor<T> B_gdmask;
    public:
        Min_op() = default;

        Tensor<T> forward(Tensor<T>& A,Tensor<T>& B) {
            if (!A_gdmask.initialize())
                A_gdmask = Tensor<T>(A.shape(),A.device(),false);
            if (!B_gdmask.initialize())
                B_gdmask = Tensor<T>(B.shape(),B.device(),false);

            Tensor<T> C = min(A,B,A_gdmask,B_gdmask);

            A.used_count()++;
            B.used_count()++;
            C.tensor_from().push_back(A);
            C.tensor_from().push_back(B);
            C.required_grad() = true;
            C.op() = this;
            return C;
        }

        std::vector<Tensor<T>> backward(std::vector<Tensor<T>>& tensor_from, Tensor<T>& grad_in) override {
            std::vector<Tensor<T>> grad_out;
            grad_out.push_back(Hadamard_op<T>::hadamard(grad_in, this->A_gdmask));
            grad_out.push_back(Hadamard_op<T>::hadamard(grad_in, this->B_gdmask));
            return grad_out;
        }

        Tensor<T> static min(Tensor<T> A,Tensor<T> B,Tensor<T> A_gdmask,Tensor<T> B_gdmask) {
#ifdef _DEBUG
            if (A.shape() != B.shape()) {
                std::cout << "Tensor_operation_error:Min_op->A.shape:" << A.shape() << "not equal to B_shape:" << B.shape() << "\n";
                throw std::invalid_argument("Tensor_operation_error:Min_op->A.shape not equal to B_shape");
            }
#endif
            Tensor<T> C(A.shape(),A.device(),false);
            cupp::Library::error_check(cupp::cu_min(A.View(),B.View(),C.View(),A_gdmask.View(),B_gdmask.View(),A.total_size(),A.dtype()));
            return C;
        }

    };

    template<typename T>
    class Max_op :public Tensor_op<T> {
    private:
        Tensor<T> A_gdmask;
        Tensor<T> B_gdmask;
    public:
        Max_op() = default;

        Tensor<T> forward(Tensor<T>& A, Tensor<T>& B) {
            if (!A_gdmask.initialize())
                A_gdmask = Tensor<T>(A.shape(), A.device(), false);
            if (!B_gdmask.initialize())
                B_gdmask = Tensor<T>(B.shape(), B.device(), false);

            Tensor<T> C = max(A, B, A_gdmask, B_gdmask);

            A.used_count()++;
            B.used_count()++;
            C.tensor_from().push_back(A);
            C.tensor_from().push_back(B);
            C.required_grad() = true;
            C.op() = this;
            return C;
        }

        std::vector<Tensor<T>> backward(std::vector<Tensor<T>>& tensor_from, Tensor<T>& grad_in) override {
            std::vector<Tensor<T>> grad_out;
            grad_out.push_back(Hadamard_op<T>::hadamard(grad_in, this->A_gdmask));
            grad_out.push_back(Hadamard_op<T>::hadamard(grad_in, this->B_gdmask));
            return grad_out;
        }

        Tensor<T> static max(Tensor<T> A, Tensor<T> B, Tensor<T> A_gdmask, Tensor<T> B_gdmask) {
#ifdef _DEBUG
            if (A.shape() != B.shape()) {
                std::cout << "Tensor_operation_error:Max_op->A.shape:" << A.shape() << "not equal to B_shape:" << B.shape() << "\n";
                throw std::invalid_argument("Tensor_operation_error:Max_op->A.shape not equal to B_shape");
            }
#endif
            Tensor<T> C(A.shape(), A.device(), false);
            cupp::Library::error_check(cupp::cu_max(A.View(), B.View(), C.View(), A_gdmask.View(), B_gdmask.View(), A.total_size(), A.dtype()));
            return C;
        }

    };

    template<typename T>
    class Slice_op : public Tensor_op<T> {
    private:
        long long slice_dim;
        long long slice_start;
        long long stride;
    public:
        Slice_op() = default;

        Tensor<T> forward(Tensor<T>& A, long long slice_dim, long long slice_start, long long slice_end, long long stride) {
            Tensor<T> B;
            B = slice(A, slice_dim, slice_start, slice_end, stride);

            this->slice_dim = slice_dim;
            this->slice_start = slice_start;
            this->stride = stride;

            A.used_count()++;
            B.tensor_from().push_back(A);
            B.required_grad() = true;
            B.op() = this;
            return B;

        }

        std::vector<Tensor<T>> backward(std::vector<Tensor<T>>& tensor_from, Tensor<T>& grad_in) override {
            std::vector<Tensor<T>> grad_out;
            grad_out.push_back(slice_backward(grad_in, tensor_from[0], slice_dim, slice_start, stride));
            return grad_out;
        }


        Tensor<T> static slice(Tensor<T> A, long long slice_dim, long long slice_start, long long slice_end, long long stride) {
#ifdef _DEBUG
            if (slice_end > A.shape()[slice_dim]) {
                std::cout << "Tensor_operation_error->Slice_op: slice_end:" << slice_end << " out of range" << "\n";
                throw std::invalid_argument("Tensor_operation_error:Slice_op: slice_end out of range");
            }
            if (slice_start < 0) {
                std::cout << "Tensor_operation_error->Slice_op: slice_start:" << slice_start << " out of range" << "\n";
                throw std::invalid_argument("Tensor_operation_error->Slice_op: slice_start should be non-negative");
            }
            if (stride <= 0) {
                std::cout << "Tensor_operation_error->Slice_op: slice_stride:" << stride << " stride should be positive" << "\n";
                throw std::invalid_argument("Tensor_operation_error->Slice_op: slice_stride should be positive");
            }
#endif
            std::vector<long long> B_shape = A.shape();
            long long slice_dim_size = (slice_end - slice_start + stride - 1) / stride;
            B_shape[slice_dim] = slice_dim_size;
            Tensor<T> B(B_shape, A.device(), false);
            long long dim_back_batch = 1;
            for (long long i = slice_dim + 1; i < A.shape().size(); i++) {
                dim_back_batch *= A.shape()[i];
            } 
            cupp::Library::error_check(cupp::cu_slice(A.View(), B.View(), B.total_size(), A.shape()[slice_dim], slice_dim_size, dim_back_batch, slice_start, stride,A.dtype()));
            return B;
        }

        Tensor<T> static slice_backward(Tensor<T> A, Tensor<T> tensor_from, long long slice_dim, long long slice_start, long long stride) {
            Tensor<T> B(tensor_from.shape(), tensor_from.device(), false);
            long long dim_back_batch = 1;
            for (long long i = slice_dim + 1; i < A.shape().size(); i++) {
                dim_back_batch *= A.shape()[i];
            }
            cupp::Library::error_check(cupp::cu_slice(A.View(), B.View(), A.total_size(), tensor_from.shape()[slice_dim], A.shape()[slice_dim], dim_back_batch, slice_start, stride,A.dtype()));
            return B;
        }

    };

    template<typename T>
    class Squeeze_op :public Tensor_op<T> {
    private:
        std::vector<long long> shape;
    public:
        Squeeze_op() = default;

        Tensor<T> forward(Tensor<T>& A) {
            Tensor<T> B = squeeze(A);
            this->shape = A.shape();

            A.used_count()++;
            B.tensor_from().push_back(A);
            B.required_grad() = true;
            B.op() = this;
            return B;
        }

        std::vector<Tensor<T>> backward(std::vector<Tensor<T>>& tensor_from, Tensor<T>& grad_in) override {
            std::vector<Tensor<T>> grad_out;
            grad_out.push_back(Reshape_op<T>::reshape(grad_in,shape));
            return grad_out;
        }

        Tensor<T> static squeeze(Tensor<T> A) {
            std::vector<long long> B_shape;
            for (long long i = 0; i < A.shape().size(); i++) {
                if (A.shape()[i] != 1)
                    B_shape.push_back(A[i]);
            }
            Tensor<T> B(B_shape,A.device(),false);
            cupp::Library::error_check(cupp::cu_Copy(A.View(),B.View(),A.total_size(),A.dtype()));
            return B;
        }
    };

    template<typename T>
    class Squeeze_with_dim_op :public Tensor_op<T> {
    private:
        std::vector<long long> shape;
    public:
        Squeeze_with_dim_op() = default;

        Tensor<T> forward(Tensor<T>& A,long long squeeze_dim) {
            Tensor<T> B = squeeze_with_dim(A,squeeze_dim);
            this->shape = A.shape();

            A.used_count()++;
            B.tensor_from().push_back(A);
            B.required_grad() = true;
            B.op() = this;
            return B;
        }

        std::vector<Tensor<T>> backward(std::vector<Tensor<T>>& tensor_from, Tensor<T>& grad_in) override {
            std::vector<Tensor<T>> grad_out;
            grad_out.push_back(Reshape_op<T>::reshape(grad_in, shape));
            return grad_out;
        }

        Tensor<T> static squeeze_with_dim(Tensor<T> A,long long squeeze_dim) {
#ifdef _DEBUG
            if (squeeze_dim >= A.shape().size() || squeeze_dim < 0) {
                std::cout << "Tensor_operation_error->squeeze_op:squeeze_dim"<< squeeze_dim << " out of range" << "\n";
                throw std::invalid_argument("Tensor_operation_error->squeeze_op:squeeze_dim out of range");
            }
            if (A.shape()[squeeze_dim] != 1) {
                std::cout << "Tensor_operation_error->squeeze_op:squeeze_dim in A_shape" << A.shape()[squeeze_dim] << " not equal to 1" << "\n";
                throw std::invalid_argument("Tensor_operation_error->squeeze_op:squeeze_dim in A_shape not equal to 1");
            }
#endif
            std::vector<long long> B_shape(A.shape().size() - 1);
            for (long long i = 0; i < squeeze_dim; i++) {
                B_shape[i] = A.shape()[i];
            }
            for (long long i = squeeze_dim; i < A.shape().size() - 1; i++) {
                B_shape[i] = A.shape()[i + 1];
            }
            Tensor<T> B(B_shape, A.device(), false);
            cupp::Library::error_check(cupp::cu_Copy(A.View(), B.View(), A.total_size(), A.dtype()));
            return B;
        }
    };


    //overloaded
    template<typename T>
    std::vector<Tensor<T>> Cos_op<T>::backward(std::vector<Tensor<T>>& tensor_from, Tensor<T>& grad_in) {
        std::vector<Tensor<T>> grad_out;
        grad_out.push_back(Hadamard_op<T>::hadamard(Opposite_op<T>::opposite(grad_in), Sin_op<T>::sin(tensor_from[0])));
        return grad_out;
    }

    template<typename T>
    std::vector<Tensor<T>> Sin_op<T>::backward(std::vector<Tensor<T>>& tensor_from, Tensor<T>& grad_in) {
        std::vector<Tensor<T>> grad_out;
        grad_out.push_back(Hadamard_op<T>::hadamard(grad_in, Cos_op<T>::cos(tensor_from[0])));
        return grad_out;
    }

    template<typename T>
    std::vector<Tensor<T>> Sum_in_dim_op<T>::backward(std::vector<Tensor<T>>& tensor_from, Tensor<T>& grad_in) {
        std::vector<Tensor<T>> grad_out;
        grad_out.push_back(Broadcast_in_dim_op<T>::broadcast_in_dim(grad_in, dim, dim_in_shape));
        return grad_out;
    }

    template<typename T>
    std::vector<Tensor<T>> Broadcast_in_dim_op<T>::backward(std::vector<Tensor<T>>& tensor_from, Tensor<T>& grad_in) {
        std::vector<Tensor<T>> grad_out;
        grad_out.push_back(Sum_in_dim_op<T>::sum_in_dim(grad_in, dim));
        return grad_out;
    }
}