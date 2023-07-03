#pragma once
#include "Operation.h"

struct GeluOperation : Operation
{
	TensorNode* input;
	TensorNode* output;

	GeluOperation(TensorNode* input, TensorNode* output)
		: input(input), output(output)
	{
		assert(input != nullptr);
		assert(output != nullptr);
		assert(input != output);
		assert(input->size == output->size);
	}

	void Forward() override
	{
		cpuGeluForward(input->size, &ONEF, input->forwardTensor, &ONEF, output->forwardTensor);
	}

	void Backward() override
	{
		cpuGeluBackward(input->size, &ONEF, output->backwardTensor, input->forwardTensor, &ONEF, input->backwardTensor);
	}

	void Update(const float* learningRate) override
	{
	}
};