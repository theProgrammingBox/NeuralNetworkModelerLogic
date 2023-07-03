#include "AddOperation.h"
#include "ReluOperation.h"
#include "AddBiasOperation.h"
#include "NeuralNetwork.h"

/*
Warnings:
- think about your alpha/beta values when using addition because if output is not initialized to 0, you will get garbage
*/

/*
TODO:
- add matmul
*/

int main()
{
	NeuralNetwork network;
	
	TensorNode* node1 = network.AddTensorNode(6);
	TensorNode* node2 = network.AddTensorNode(6);

	network.AddOperation(new ReluOperation(node1, node2));
	network.AddOperation(new AddOperation(node1, node2));
	network.AddOperation(new AddBiasOperation(node2));

	for (int i = 0; i < 6; i++)
		node1->forwardTensor[i] = i - 3;
	
	network.Forward();
	
	node1->PrintForwardTensor();
	node2->PrintForwardTensor();
	
	return 0;
}