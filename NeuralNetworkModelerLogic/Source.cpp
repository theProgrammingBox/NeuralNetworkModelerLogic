#include "AddOperation.h"
#include "ReluOperation.h"
#include "AddBiasOperation.h"
#include "MultiplyWeightOperation.h"
#include "NeuralNetwork.h"

/*
Warnings:
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
	TensorNode* product = network.AddTensorNode(8);
	TensorNode* relu = network.AddTensorNode(8);
	TensorNode* product2 = network.AddTensorNode(8);

	network.AddOperation(new MultiplyWeightOperation(input, product));
	network.AddOperation(new AddBiasOperation(product));
	network.AddOperation(new ReluOperation(product, relu));
	network.AddOperation(new MultiplyWeightOperation(relu, product2));
	network.AddOperation(new AddOperation(input, product2));

	network.ZeroForwardTensors();
	for (int i = 0; i < 8; i++)
		input->forwardTensor[i] = i;
	
	network.Forward();
	
	input->PrintForward("input");
	product->PrintForward("product");
	relu->PrintForward("relu");
	product2->PrintForward("product2");
	
	return 0;
}