#pragma once
#include "Operation.h"

struct ConcatOperation : Operation
{
	TensorNode* input1;
	TensorNode* input2;
	TensorNode* output;

	ConcatOperation(TensorNode* input1, TensorNode* input2, TensorNode* output)
		: input1(input1), input2(input2), output(output)
	{
		assert(input1 != nullptr);
		assert(input2 != nullptr);
		assert(output != nullptr);
		assert(input1 != output);
		assert(input2 != output);
		assert(input1->size + input2->size == output->size);
	}

	~ConcatOperation()
	{
	}

	void Forward() override
	{
	}

	void Backward() override
	{
	}

	void Update(const float* learningRate) override
	{
	}

	void PrintParam() const override
	{
	}
};