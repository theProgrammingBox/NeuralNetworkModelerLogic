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
};