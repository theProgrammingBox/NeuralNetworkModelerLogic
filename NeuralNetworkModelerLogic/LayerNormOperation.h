#pragma once
#include "Operation.h"

struct LayerNormOperation : Operation
{
	TensorNode* input;
	TensorNode* output;
	float invSize;
	float sqrtSize;
	float mean;
	float invStd;

	LayerNormOperation(TensorNode* input, TensorNode* output)
		: input(input), output(output)
	{
		assert(input != nullptr);

		invSize = 1.0f / input->size;
		sqrtSize = sqrtf(input->size);
	}

	~LayerNormOperation()
	{
	}

	void Forward() override
	{
		mean = 0.0f;
		for (int i = 0; i < input->size; i++)
			mean += input->forwardTensor[i];
		mean *= invSize;
		
		invStd = 0.0f;
		float x;
		for (int i = 0; i < input->size; i++)
		{
			x = input->forwardTensor[i] - mean;
			invStd += x * x;
		}
		invStd = InvSqrt(invStd + 1e-16f) * sqrtSize;

		for (int i = 0; i < input->size; i++)
			output->forwardTensor[i] = (input->forwardTensor[i] - mean) * invStd;
	}

	void Backward() override
	{
		float dMean = 0.0f;
		float dInvStd = 0.0f;
		
		for (int i = 0; i < input->size; i++)
		{
			dMean += output->backwardTensor[i];
			dInvStd += output->backwardTensor[i] * output->forwardTensor[i];
		}
		dMean *= -invStd * invSize;
		dInvStd *= -invStd * invStd * invSize;
		
		for (int i = 0; i < input->size; i++)
			input->backwardTensor[i] = output->backwardTensor[i] * invStd + dMean + dInvStd * (input->forwardTensor[i] - mean);
	}

	void Update(const float* learningRate) override
	{
	}

	void PrintParam() const override
	{
	}
};