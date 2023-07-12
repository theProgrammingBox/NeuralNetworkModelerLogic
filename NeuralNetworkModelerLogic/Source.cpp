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
- visulize performance
- try other initialization methods
- add layer norm
- understand layernorm / test with new addition backprop for stability
*/

int main()
{
	srand(time(NULL));
	
	const float LEARNING_RATE = 0.01f;
	const int BATCH_SIZE = 32;
	const int EPISODES = 10000;

	float UPDATE_RATE = LEARNING_RATE * InvSqrt(BATCH_SIZE);

	NeuralNetwork network;

	TensorNode* input = network.AddTensorNode(new TensorNode("input", 16));

	TensorNode* product1 = network.AddTensorNode(new TensorNode("product1", 32));
	TensorNode* gelu1 = network.AddTensorNode(new TensorNode("gelu1", 32));
	TensorNode* product2 = network.AddTensorNode(new TensorNode("product2", 16));

	TensorNode* product3 = network.AddTensorNode(new TensorNode("product3", 32));
	TensorNode* gelu2 = network.AddTensorNode(new TensorNode("gelu2", 32));
	TensorNode* product4 = network.AddTensorNode(new TensorNode("product3", 16));

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
			uint8_t b = rand();
			uint8_t c = a + b;
			
			network.ZeroForward();
			for (int i = 0; i < 8; i++)
				input->forwardTensor[i] = (a >> i) & 1;
			for (int i = 0; i < 8; i++)
				input->forwardTensor[i + 8] = (b >> i) & 1;

			network.Forward();

			network.ZeroBackward();
			for (int i = 0; i < 8; i++)
				output->backwardTensor[i] = ((c >> i) & 1) - output->forwardTensor[i];

			network.Backward();
		}
		network.Update(&UPDATE_RATE);
	}

	network.PrintParam();
	printf("\n");

	network.PrintForward();
	printf("\n");

	network.PrintBackward();

	return 0;
}