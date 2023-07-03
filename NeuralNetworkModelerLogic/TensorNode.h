#pragma once
#include "Header.h"

struct TensorNode
{
	int size;
	float* forwardTensor;
	float* backwardTensor;

	TensorNode(int size)
		: size(size)
	{
		forwardTensor = new float[size];
		backwardTensor = new float[size];
	}

	~TensorNode()
	{
		delete[] forwardTensor;
		delete[] backwardTensor;
	}

	void PrintForwardTensor() const
	{
		PrintMatrixf32(forwardTensor, 1, size, "Forward Tensor");
	}

	void PrintBackwardTensor() const
	{
		PrintMatrixf32(backwardTensor, 1, size, "Backward Tensor");
	}
};