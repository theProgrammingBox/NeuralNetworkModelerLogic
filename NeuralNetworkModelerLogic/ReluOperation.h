#pragma once
#include "Operation.h"

struct ReluOperation : Operation
{
	TensorNode* input;
	TensorNode* output;

	ReluOperation(TensorNode* input, TensorNode* output)
		: input(input), output(output)
	{
		assert(input != nullptr);
		assert(output != nullptr);
		assert(input != output);
		assert(input->size == output->size);
	}

	void Forward() override
	{
		cpuReluForward(input->size, &ONEF, input->forwardTensor, &ONEF, output->forwardTensor);
	}

	void Backward() override
	{
		cpuReluBackward(input->size, &ONEF, output->backwardTensor, input->forwardTensor, &ONEF, input->backwardTensor);
	}

	void Update(const float* learningRate) override
	{
	}

	void PrintParam() const override
	{
	}
};