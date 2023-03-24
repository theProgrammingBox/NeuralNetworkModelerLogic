#pragma once
#include "Matrix.h"

class OperationComponent
{
public:
	OperationComponent() = default;
	virtual ~OperationComponent() = default;

	virtual void SetInputMatrix(uint8_t idx, Matrix* matrix) = 0;
	virtual void SetOutputMatrix(uint8_t idx, Matrix* matrix) = 0;

protected:
	uint8_t inputCount;
	uint8_t outputCount;
	Matrix** inputMatrices;
	Matrix** outputMatrices;
};