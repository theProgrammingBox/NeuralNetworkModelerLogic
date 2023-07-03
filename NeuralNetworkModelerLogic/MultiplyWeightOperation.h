#pragma once
#include "Operation.h"

struct MultiplyWeightOperation : Operation
{
	TensorNode* input;
	TensorNode* output;
	TensorNode* weight;

	MultiplyWeightOperation(TensorNode* input, TensorNode* output)
		: input(input), output(output)
	{
		assert(input != nullptr);
		assert(output != nullptr);
		assert(input != output);
		assert(input->height == output->height);
		
		weight = new TensorNode(output->width, input->width);
		
		weight->ZeroForward();
		for (int i = 0; i < std::min(weight->width, weight->height); i++)
			weight->forwardTensor[i * weight->width + i] = 1;
	}

	~MultiplyWeightOperation()
	{
		delete weight;
	}

	void Forward() override
	{
		cpuSgemmStridedBatched(
			false, false,
			output->width, input->height, input->width,
			&ONEF,
			weight->forwardTensor, output->width, output->width * input->height,
			input->forwardTensor, input->width, input->width * input->height,
			&ONEF,
			output->forwardTensor, output->width, output->width * input->height,
			1);
	}
};