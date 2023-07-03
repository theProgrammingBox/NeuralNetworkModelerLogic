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
		
		bias = new TensorNode("Bias", input->size);
		//bias->ZeroForward();
		for (int i = 0; i < bias->size; i++)
			bias->forwardTensor[i] = 1.0f;
		bias->ZeroBackward();
	}

	~AddBiasOperation()
	{
		delete bias;
	}

	void Forward() override
	{
		cpuSaxpy(input->size, &ONEF, bias->forwardTensor, 1, input->forwardTensor, 1);
	}

	void Backward() override
	{
		cpuSaxpy(input->size, &ONEF, input->backwardTensor, 1, bias->backwardTensor, 1);
	}

	void Update(const float* learningRate) override
	{
		cpuSaxpy(input->size, learningRate, bias->backwardTensor, 1, bias->forwardTensor, 1);
		bias->ZeroBackward();
	}

	void PrintParam() const override
	{
		bias->PrintForward();
	}
};