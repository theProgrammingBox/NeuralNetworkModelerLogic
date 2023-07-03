#include "AddOperation.h"
#include "ReluOperation.h"
#include "GeluOperation.h"
#include "AddBiasOperation.h"
#include "MultiplyWeightOperation.h"
#include "NeuralNetwork.h"

/*
Warnings:
*/

/*
TODO:
- add layer norm
- add the operation type in the label
*/

int main()
{
	const float LEARNING_RATE = 0.002f;
	const int BATCH_SIZE = 1;
	const int EPISODES = 10;

	float UPDATE_RATE = LEARNING_RATE / BATCH_SIZE;
	
	NeuralNetwork network;
	
	TensorNode* input = network.AddTensorNode(new TensorNode("input", 8));
	TensorNode* product = network.AddTensorNode(new TensorNode("product", 8));
	TensorNode* relu = network.AddTensorNode(new TensorNode("relu", 8));
	TensorNode* product2 = network.AddTensorNode(new TensorNode("product2", 8));

	network.AddOperation(new MultiplyWeightOperation(input, product));
	network.AddOperation(new AddBiasOperation(product));
	network.AddOperation(new GeluOperation(product, relu));
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
				product2->backwardTensor[i] = -i - product2->forwardTensor[i];

			network.Backward();
		}
		network.Update(&UPDATE_RATE);
	}
	
	network.PrintForward();
	printf("\n");

	network.PrintBackward();
	printf("\n");
	
	network.PrintParam();
	
	return 0;
}