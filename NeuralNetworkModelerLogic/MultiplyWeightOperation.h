#pragma once
#include "Operation.h"

struct MultiplyWeightOperation : Operation
{
	bool transposeA;
	bool transposeB;
	int inputHeight;
	int inputWidth;
	int outputWidth;
	const float* alpha;
	const float* beta;
	
	TensorNode* input;
	TensorNode* output;
	TensorNode* weight;

	MultiplyWeightOperation(TensorNode* input, TensorNode* output, bool transposeA = false, bool transposeB = false, int inputHeight = 0, int inputWidth = 0, int outputWidth = 0, const float* alpha = &ONEF, const float* beta = &ZEROF)
		: input(input), output(output), transposeA(transposeA), transposeB(transposeB), inputHeight(inputHeight), inputWidth(inputWidth), outputWidth(outputWidth), alpha(alpha), beta(beta)
	{
		if (inputHeight == 0)
			this->inputHeight = input->height;
		if (inputWidth == 0)
			this->inputWidth = input->width;
		if (outputWidth == 0)
			this->outputWidth = output->width;
		
		weight = new TensorNode(this->outputWidth, this->inputWidth);
		weight->ZeroForwardTensor();
		// identity matrix
		for (int i = 0; i < this->outputWidth; i++)
			weight->forwardTensor[i * this->inputWidth + i] = 1;
		weight->ZeroBackwardTensor();
	}

	~MultiplyWeightOperation()
	{
		delete weight;
	}

	void Forward() override
	{
		cpuSgemmStridedBatched(
			transposeA, transposeB,
			outputWidth, inputHeight, inputWidth,
			alpha,
			weight->forwardTensor, outputWidth, outputWidth * inputWidth,
			input->forwardTensor, inputWidth, inputWidth * inputHeight,
			beta,
			output->forwardTensor, outputWidth, outputWidth * inputHeight,
			1);
	}
};