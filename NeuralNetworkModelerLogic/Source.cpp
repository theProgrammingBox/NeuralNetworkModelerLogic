#include <iostream>

struct Tensor
{
};

struct Operation
{
};

struct Concatenate : Operation
{
	Concatenate(Tensor* v1, Tensor* v2, Tensor* v3)
	{
	}
};

struct RecursiveLink : Operation
{
	RecursiveLink(Tensor* v1, Tensor* v2)
	{
	}
};

struct MatMul : Operation
{
	MatMul(Tensor* v1, Tensor* v2, Tensor* v3)
	{
	}
};

struct Relu : Operation
{
	Relu(Tensor* v1, Tensor* v2)
	{
	}
};

struct MatAdd : Operation
{
	MatAdd(Tensor* v1, Tensor* v2, Tensor* v3)
	{
	}
};

int main()
{
	auto input = new Tensor();
	auto hidden = new Tensor();
	auto concat = new Tensor();
	auto weight1 = new Tensor();
	auto product = new Tensor();
	auto relu = new Tensor();
	auto weight2 = new Tensor();
	auto output = new Tensor();
	auto weight3 = new Tensor();
	auto presum = new Tensor();
	auto newHidden = new Tensor();

	auto recursive = new RecursiveLink(newHidden, hidden);
	auto concatenate = new Concatenate(hidden, input, concat);
	auto matmul1 = new MatMul(concat, weight1, product);
	auto relu1 = new Relu(product, relu);
	auto matmul2 = new MatMul(relu, weight2, output);
	auto matmul3 = new MatMul(relu, weight3, presum);
	auto matadd = new MatAdd(presum, hidden, newHidden);

	return 0;
}