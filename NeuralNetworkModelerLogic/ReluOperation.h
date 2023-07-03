#pragma once
#include "Operation.h"

struct ReluOperation : Operation
{
	int size;
	const float* alpha;
	const float* beta;
	TensorNode* input;
	TensorNode* output;

	ReluOperation(TensorNode* input, TensorNode* output = nullptr, int size = 0, const float* alpha = &ONEF, const float* beta = &ZEROF)
		: input(input), output(output), size(size), alpha(alpha), beta(beta)
	{
		if (size == 0)
			this->size = input->size;
		if (output == nullptr)
			this->output = input;
	}

	void Forward() override
	{
		cpuReluForward(size, alpha, input->forwardTensor, beta, output->forwardTensor);
	}
};