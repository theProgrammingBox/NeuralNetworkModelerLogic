#pragma once
#include "Operation.h"

struct NeuralNetwork
{
	std::vector<TensorNode*> nodes;
	std::vector<Operation*> operations;

	~NeuralNetwork()
	{
		for (TensorNode* node : nodes)
			delete node;
		for (Operation* operation : operations)
			delete operation;
	}

	TensorNode* AddTensorNode(int size)
	{
		TensorNode* node = new TensorNode(size);
		nodes.emplace_back(node);
		return node;
	}

	Operation* AddOperation(Operation* operation)
	{
		operations.emplace_back(operation);
		return operation;
	}

	void ZeroForward()
	{
		for (TensorNode* node : nodes)
			node->ZeroForward();
	}

	void ZeroBackward()
	{
		for (TensorNode* node : nodes)
			node->ZeroBackward();
	}

	void Forward()
	{
		for (Operation* operation : operations)
			operation->Forward();
	}

	void Backward()
	{
		for (auto it = operations.rbegin(); it != operations.rend(); ++it)
			(*it)->Backward();
	}

	void Update(const float* learningRate)
	{
		for (Operation* operation : operations)
			operation->Update(learningRate);
	}
};