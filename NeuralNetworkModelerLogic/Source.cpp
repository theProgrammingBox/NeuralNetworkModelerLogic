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
	srand(time(NULL));
	
	const float LEARNING_RATE = 0.003f;
	const int BATCH_SIZE = 8;
	const int EPISODES = 200;

	float UPDATE_RATE = LEARNING_RATE * InvSqrt(BATCH_SIZE);

	NeuralNetwork network;

	TensorNode* input = network.AddTensorNode(new TensorNode("input", 8));

	TensorNode* product1 = network.AddTensorNode(new TensorNode("product1", 8));
	TensorNode* gelu1 = network.AddTensorNode(new TensorNode("gelu1", 8));
	TensorNode* product2 = network.AddTensorNode(new TensorNode("product2", 8));

	TensorNode* product3 = network.AddTensorNode(new TensorNode("product3", 8));
	TensorNode* gelu2 = network.AddTensorNode(new TensorNode("gelu2", 8));
	TensorNode* product4 = network.AddTensorNode(new TensorNode("product3", 8));

	TensorNode* output = network.AddTensorNode(new TensorNode("product5", 8));

	network.AddOperation(new MultiplyWeightOperation(input, product1));
	network.AddOperation(new AddBiasOperation(product1));
	network.AddOperation(new GeluOperation(product1, gelu1));
	network.AddOperation(new MultiplyWeightOperation(gelu1, product2));
	network.AddOperation(new AddOperation(input, product2));

	network.AddOperation(new MultiplyWeightOperation(product2, product3));
	network.AddOperation(new AddBiasOperation(product3));
	network.AddOperation(new GeluOperation(product3, gelu2));
	network.AddOperation(new MultiplyWeightOperation(gelu2, product4));
	network.AddOperation(new AddOperation(product2, product4));
	
	network.AddOperation(new MultiplyWeightOperation(product4, output));

	for (int episode = 0; episode < EPISODES; episode++)
	{
		for (int batch = 0; batch < BATCH_SIZE; batch++)
		{
			uint8_t a = rand();
			
			network.ZeroForward();
			for (int i = 0; i < 8; i++)
				input->forwardTensor[i] = a >> i & 1;

			network.Forward();

			network.ZeroBackward();
			for (int i = 0; i < 8; i++)
				output->backwardTensor[i] = input->forwardTensor[i] - output->forwardTensor[i];

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