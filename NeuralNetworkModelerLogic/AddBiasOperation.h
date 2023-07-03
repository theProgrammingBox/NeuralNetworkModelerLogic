#pragma once
#include "Operation.h"

struct AddBiasOperation : Operation
{
	int size;
	const float* alpha;
	int incx;
	int incy;
	TensorNode* input;
	float* bias;

	AddBiasOperation(TensorNode* input, int size = 0, const float* alpha = &ONEF, int incx = 1, int incy = 1)
		: input(input), size(size), alpha(alpha), incx(incx), incy(incy)
	{
		if (size == 0)
			this->size = input->size;
		bias = new float[input->size];
		//memset(bias, 0, sizeof(float) * input->size);
		for (int i = 0; i < input->size; i++)
			bias[i] = 1.0f;
	}

	~AddBiasOperation()
	{
		delete[] bias;
	}

	void Forward() override
	{
		cpuSaxpy(size, alpha, bias, incx, input->forwardTensor, incy);
	}
};