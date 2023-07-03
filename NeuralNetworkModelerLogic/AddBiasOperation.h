#pragma once
#include "Operation.h"

struct AddBiasOperation : Operation
{
	int size;
	const float* alpha;
	int incx;
	int incy;
	TensorNode* input;
	TensorNode* bias;

	AddBiasOperation(TensorNode* input, int size = 0, const float* alpha = &ONEF, int incx = 1, int incy = 1)
		: input(input), size(size), alpha(alpha), incx(incx), incy(incy)
	{
		if (size == 0)
			this->size = input->size;
		bias = new TensorNode(this->size);
		bias->ZeroForwardTensor();
		for (int i = 0; i < this->size; i++)
			bias->forwardTensor[i] = 1;
		bias->ZeroBackwardTensor();
	}

	~AddBiasOperation()
	{
		delete bias;
	}

	void Forward() override
	{
		cpuSaxpy(size, alpha, bias->forwardTensor, incx, input->forwardTensor, incy);
	}
};