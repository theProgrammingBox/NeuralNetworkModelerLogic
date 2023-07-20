#pragma once
#include "Operation.h"

struct MultiplyWeightOperation : Operation
{
	TensorNode* input;
	TensorNode* output;
	TensorNode* weight;

	MultiplyWeightOperation(TensorNode* input, TensorNode* output)
		: input(input), output(output)
	{
		assert(input != nullptr);
		assert(output != nullptr);
		assert(input != output);
		assert(input->height == output->height);
		
		weight = new TensorNode("weight", output->width, input->width);
		/*weight->ZeroForward();
		for (int i = 0; i < std::min(input->width, output->width); i++)
			weight->forwardTensor[i * output->width + i] = 1.0f;*/
		for (int i = 0; i < weight->size; i++)
			weight->forwardTensor[i] = RandomFloat() - 0.5f;
		weight->ZeroBackward();
	}

	~MultiplyWeightOperation()
	{
		delete weight;
	}

	void Forward() override
	{
		cpuSgemmStridedBatched(
			false, false,
			output->width, input->height, input->width,
			&ONEF,
			weight->forwardTensor, output->width, output->width * input->width,
			input->forwardTensor, input->width, input->width * input->height,
			&ONEF,
			output->forwardTensor, output->width, output->width * input->height,
			1);
	}

	void Backward() override
	{
		cpuSgemmStridedBatched(
			false, true,
			output->width, input->width, input->height,
			&ONEF,
			output->backwardTensor, output->width, output->width * input->height,
			input->forwardTensor, input->width, input->width * input->height,
			&ONEF,
			weight->backwardTensor, output->width, output->width * input->width,
			1);
		cpuSgemmStridedBatched(
			true, false,
			input->width, input->height, output->width,
			&ONEF,
			weight->forwardTensor, output->width, output->width * input->width,
			output->backwardTensor, output->width, output->width * input->height,
			&ONEF,
			input->backwardTensor, input->width, input->width * input->height,
			1);
	}

	void Update(const float* learningRate) override
	{
		cpuSaxpy(weight->size, learningRate, weight->backwardTensor, 1, weight->forwardTensor, 1);
		weight->ZeroBackward();
	}

	void PrintParam() const override
	{
		weight->PrintForward();
	}
};