#include "AddOperation.h"
#include "ReluOperation.h"
#include "AddBiasOperation.h"
#include "MultiplyWeightOperation.h"
#include "NeuralNetwork.h"

/*
Warnings:
- be careful about your alpha/beta values when they default to 1 because if output is not initialized to 0 or a result of another operation, you will get garbage
-- compiler can check if there is an operation on output node with an alpha/beta of 1 before hand
*/

/*
TODO:
- work on backward
- add matmul
*/

int main()
{
	NeuralNetwork network;
	
	TensorNode* input = network.AddTensorNode(8);
	TensorNode* productAndRelu = network.AddTensorNode(8);
	TensorNode* productAndSum = network.AddTensorNode(8);

	network.AddOperation(new MultiplyWeightOperation(input, productAndRelu));
	network.AddOperation(new AddBiasOperation(productAndRelu));
	network.AddOperation(new ReluOperation(productAndRelu));
	network.AddOperation(new MultiplyWeightOperation(productAndRelu, productAndSum));
	network.AddOperation(new AddOperation(input, productAndSum));

	for (int i = 0; i < 8; i++)
		input->forwardTensor[i] = i;
	
	network.Forward();
	
	input->PrintForwardTensor();
	productAndRelu->PrintForwardTensor();
	productAndSum->PrintForwardTensor();
	
	return 0;
}