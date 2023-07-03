#pragma once
#include "Operation.h"

struct AddBiasOperation : Operation
{
	TensorNode* input;
	TensorNode* bias;

	AddBiasOperation(TensorNode* input)
		: input(input)
	{
		assert(input != nullptr);
		bias = new TensorNode(input->size);
		
		for (int i = 0; i < bias->size; i++)
			bias->forwardTensor[i] = 1;
	}

	~AddBiasOperation()
	{
		delete bias;
	}

	void Forward() override
	{
		cpuSaxpy(input->size, &ONEF, bias->forwardTensor, 1, input->forwardTensor, 1);
	}
};