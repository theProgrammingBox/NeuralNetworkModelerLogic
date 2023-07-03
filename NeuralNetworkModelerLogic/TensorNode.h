#pragma once
#include "Header.h"

struct TensorNode
{
	int width;
	int height;
	int channels;
	int batches;

	const char* label;
	
	int size;
	float* forwardTensor;
	float* backwardTensor;

	TensorNode(const char* label, int width, int height = 1, int channels = 1, int batches = 1)
		: label(label), width(width), height(height), channels(channels), batches(batches)
	{
		size = width * height * channels * batches;
		forwardTensor = new float[size];
		backwardTensor = new float[size];
	}

	~TensorNode()
	{
		delete[] forwardTensor;
		delete[] backwardTensor;
	}

	void ZeroForward()
	{
		memset(forwardTensor, 0, sizeof(float) * size);
	}

	void ZeroBackward()
	{
		memset(backwardTensor, 0, sizeof(float) * size);
	}

	void PrintForward() const
	{
		PrintMatrixf32(forwardTensor, width, height, label);
	}

	void PrintBackward() const
	{
		PrintMatrixf32(backwardTensor, width, height, label);
	}
};