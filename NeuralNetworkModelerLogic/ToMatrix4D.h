#pragma once
#include "OperationComponent.h"

class ToMatrix4D : public OperationComponent
{
public:
	ToMatrix4D()
	{
		inputCount = 1;
		outputCount = 1;
		inputMatrices = new Matrix * [inputCount];
		outputMatrices = new Matrix * [outputCount];
	}

	virtual void SetInputMatrix(uint8_t idx, Matrix* matrix) override
	{
		if (idx < inputCount)
		{
			inputMatrices[idx] = matrix;
		}
	}

	virtual void SetOutputMatrix(uint8_t idx, Matrix* matrix) override
	{
		if (idx < outputCount)
		{
			outputMatrices[idx] = matrix;
		}
	}
};