#pragma once
#include "Operation.h"

struct AddOperation : Operation
{
	int size;
	const float* alpha;
	int incx;
	int incy;
	TensorNode* input1;
	TensorNode* input2;

	AddOperation(TensorNode* input1, TensorNode* input2, int size = 0, const float* alpha = &ONEF, int incx = 1, int incy = 1)
		: input1(input1), input2(input2), size(size), alpha(alpha), incx(incx), incy(incy)
	{
		if (size == 0)
			this->size = input1->size;
	}

	void Forward() override
	{
		cpuSaxpy(size, alpha, input1->forwardTensor, incx, input2->forwardTensor, incy);
	}
};