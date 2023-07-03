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
*/

int main()
{
	const float LEARNING_RATE = 0.002f;
	const int BATCH_SIZE = 1;
	const int EPISODES = 10;

	float UPDATE_RATE = LEARNING_RATE / BATCH_SIZE;
	
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

	for (int episode = 0; episode < EPISODES; episode++)
	{
		for (int batch = 0; batch < BATCH_SIZE; batch++)
		{
			network.ZeroForward();
			for (int i = 0; i < 8; i++)
				input->forwardTensor[i] = i;

			network.Forward();

			network.ZeroBackward();
			for (int i = 0; i < 8; i++)
				product2->backwardTensor[i] = 8 - product2->forwardTensor[i];

			network.Backward();
		}
		network.Update(&UPDATE_RATE);
	}

	input->PrintForward("input");
	product->PrintForward("product");
	relu->PrintForward("relu");
	product2->PrintForward("product2");
	printf("\n");

	product2->PrintBackward("product2 Gradient");
	relu->PrintBackward("relu Gradient");
	product->PrintBackward("product Gradient");
	input->PrintBackward("input Gradient");
	
	return 0;
}