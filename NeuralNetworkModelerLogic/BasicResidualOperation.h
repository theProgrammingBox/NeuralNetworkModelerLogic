#pragma once
#include "Operation.h"

#include "AddOperation.h"
#include "GeluOperation.h"
#include "AddBiasOperation.h"
#include "MultiplyWeightOperation.h"

struct BasicResidualOperation : Operation
{
	TensorNode* input;
	TensorNode* product;
	TensorNode* gelu;
	TensorNode* output;

	Operation* MultiplyWeightOp;
	Operation* AddBiasOp;
	Operation* GeluOp;
	Operation* MultiplyWeight2Op;
	Operation* AddOp;

	BasicResidualOperation(TensorNode* input, TensorNode* output)
		: input(input), output(output)
	{
		assert(input != nullptr);
		assert(output != nullptr);
		assert(input != output);
		assert(input->size == output->size);
		
		product = new TensorNode("Product", input->size);
		gelu = new TensorNode("Gelu", input->size);
		
		MultiplyWeightOp = new MultiplyWeightOperation(input, product);
		AddBiasOp = new AddBiasOperation(product);
		GeluOp = new GeluOperation(product, gelu);
		MultiplyWeight2Op = new MultiplyWeightOperation(gelu, output);
		AddOp = new AddOperation(input, output);
	}

	~BasicResidualOperation()
	{
		delete product;
		delete gelu;
		
		delete MultiplyWeightOp;
		delete AddBiasOp;
		delete GeluOp;
		delete MultiplyWeight2Op;
		delete AddOp;
	}

	void Forward() override
	{
		MultiplyWeightOp->Forward();
		AddBiasOp->Forward();
		GeluOp->Forward();
		MultiplyWeight2Op->Forward();
		AddOp->Forward();
	}

	void Backward() override
	{
		AddOp->Backward();
		MultiplyWeight2Op->Backward();
		GeluOp->Backward();
		AddBiasOp->Backward();
		MultiplyWeightOp->Backward();
	}

	void Update(const float* learningRate) override
	{
		MultiplyWeightOp->Update(learningRate);
		AddBiasOp->Update(learningRate);
		MultiplyWeight2Op->Update(learningRate);
	}

	

	void PrintParam() const override
	{
		MultiplyWeightOp->PrintParam();
		AddBiasOp->PrintParam();
		MultiplyWeight2Op->PrintParam();
	}
};