#pragma once
#include "Operation.h"

struct SigmoidOperation : Operation
{
	TensorNode* input;
	TensorNode* output;

	SigmoidOperation(TensorNode* input, TensorNode* output)
		: input(input), output(output)
	{
		assert(input != nullptr);
		assert(output != nullptr);
		assert(input != output);
		assert(input->size == output->size);
	}

	~SigmoidOperation()
	{
	}

	void Forward() override
	{
		cpuSigmoidForward(input->size, &ONEF, input->forwardTensor, &ONEF, output->forwardTensor);
	}

	void Backward() override
	{
		cpuSigmoidBackward(input->size, &ONEF, output->backwardTensor, input->forwardTensor, &ONEF, input->backwardTensor);
	}

	void Update(const float* learningRate) override
	{
	}

	void PrintParam() const override
	{
	}
};