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

	void Forward()
	{
		for (Operation* operation : operations)
			operation->Forward();
	}
};