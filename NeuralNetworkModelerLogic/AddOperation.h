#pragma once
#include "Operation.h"

struct AddOperation : Operation
{
	TensorNode* input;
	TensorNode* output;

	AddOperation(TensorNode* input, TensorNode* output)
		: input(input), output(output)
	{
		assert(input != nullptr);
		assert(output != nullptr);
		assert(input != output);
		assert(input->size == output->size);
	}

	void Forward() override
	{
		cpuSaxpy(input->size, &ONEF, input->forwardTensor, 1, output->forwardTensor, 1);
	}

	void Backward() override
	{
		cpuSaxpy(input->size, &ONEF, output->backwardTensor, 1, input->backwardTensor, 1);
	}

	void Update(const float* learningRate) override
	{
	}
	
	void PrintParam() const override
	{
	}
};