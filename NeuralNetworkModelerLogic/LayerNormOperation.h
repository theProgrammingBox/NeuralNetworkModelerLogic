#pragma once
#include "Operation.h"

struct LayerNormOperation : Operation
{
	TensorNode* input;
	float negInvSize;
	float mean;
	float variance;

	LayerNormOperation(TensorNode* input)
		: input(input)
	{
		assert(input != nullptr);

		negInvSize = -1.0f / input->size;
	}

	~LayerNormOperation()
	{
	}

	void Forward() override
	{
		mean = 0.0f;
		variance = 0.0f;
		for (int i = 0; i < input->size; i++)
			mean += input->forwardTensor[i];
		mean *= negInvSize;
		
		for (int i = 0; i < input->size; i++)
		{
			float x = input->forwardTensor[i] + mean;
			variance += x * x;
		}
		variance = InvSqrt(variance + 1e-16f);

		for (int i = 0; i < input->size; i++)
			input->forwardTensor[i] = (input->forwardTensor[i] + mean) * variance;
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