#include "AddOperation.h"
#include "ReluOperation.h"
#include "GeluOperation.h"
#include "AddBiasOperation.h"
#include "MultiplyWeightOperation.h"
#include "NeuralNetwork.h"

/*
TODO:
- test other architectures
- try other initialization methods
- add layer norm
- understand layernorm / test with new addition backprop for stability
*/

int main()
{
	srand(time(NULL));
	
	const float LEARNING_RATE = 0.006f;
	const int BATCH_SIZE = 32;
	const int EPISODES = 100000;
	const int LOG_LENGTH = 56;
	const int EPISODES_PER_PRINT = EPISODES / LOG_LENGTH;
	
	float UPDATE_RATE = LEARNING_RATE * InvSqrt(BATCH_SIZE);

	NeuralNetwork network;

	/*TensorNode* input = network.AddTensorNode(new TensorNode("input", 16));

	TensorNode* product1 = network.AddTensorNode(new TensorNode("product1", 32));
	TensorNode* gelu1 = network.AddTensorNode(new TensorNode("gelu1", 32));
	TensorNode* product2 = network.AddTensorNode(new TensorNode("product2", 16));

	TensorNode* product3 = network.AddTensorNode(new TensorNode("product3", 32));
	TensorNode* gelu2 = network.AddTensorNode(new TensorNode("gelu2", 32));
	TensorNode* product4 = network.AddTensorNode(new TensorNode("product4", 16));

	TensorNode* product5 = network.AddTensorNode(new TensorNode("product5", 32));
	TensorNode* gelu3 = network.AddTensorNode(new TensorNode("gelu3", 32));
	TensorNode* product6 = network.AddTensorNode(new TensorNode("product6", 16));

	TensorNode* product7 = network.AddTensorNode(new TensorNode("product7", 32));
	TensorNode* gelu4 = network.AddTensorNode(new TensorNode("gelu4", 32));
	TensorNode* product8 = network.AddTensorNode(new TensorNode("product8", 16));

	TensorNode* output = network.AddTensorNode(new TensorNode("output", 8));*/

	TensorNode* input = network.AddTensorNode(new TensorNode("input", 16));
	
	TensorNode* product1 = network.AddTensorNode(new TensorNode("product1", 32));
	TensorNode* activation1 = network.AddTensorNode(new TensorNode("activation1", 32));
	
	TensorNode* product2 = network.AddTensorNode(new TensorNode("product2", 16));
	TensorNode* activation2 = network.AddTensorNode(new TensorNode("activation2", 16));

	TensorNode* product3 = network.AddTensorNode(new TensorNode("product3", 16));
	TensorNode* activation3 = network.AddTensorNode(new TensorNode("activation3", 16));

	TensorNode* product4 = network.AddTensorNode(new TensorNode("product4", 16));
	TensorNode* activation4 = network.AddTensorNode(new TensorNode("activation4", 16));
	
	TensorNode* output = network.AddTensorNode(new TensorNode("output", 8));

	/*network.AddOperation(new MultiplyWeightOperation(input, product1));
	network.AddOperation(new AddBiasOperation(product1));
	network.AddOperation(new GeluOperation(product1, gelu1));
	network.AddOperation(new MultiplyWeightOperation(gelu1, product2));
	network.AddOperation(new AddOperation(input, product2));

	network.AddOperation(new MultiplyWeightOperation(product2, product3));
	network.AddOperation(new AddBiasOperation(product3));
	network.AddOperation(new GeluOperation(product3, gelu2));
	network.AddOperation(new MultiplyWeightOperation(gelu2, product4));
	network.AddOperation(new AddOperation(product2, product4));

	network.AddOperation(new MultiplyWeightOperation(product4, product5));
	network.AddOperation(new AddBiasOperation(product5));
	network.AddOperation(new GeluOperation(product5, gelu3));
	network.AddOperation(new MultiplyWeightOperation(gelu3, product6));

	network.AddOperation(new MultiplyWeightOperation(product6, product7));
	network.AddOperation(new AddBiasOperation(product7));
	network.AddOperation(new GeluOperation(product7, gelu4));
	network.AddOperation(new MultiplyWeightOperation(gelu4, product8));
	
	network.AddOperation(new MultiplyWeightOperation(product8, output));*/

	network.AddOperation(new MultiplyWeightOperation(input, product1));
	network.AddOperation(new ReluOperation(product1, activation1));
	network.AddOperation(new AddBiasOperation(activation1));
	
	network.AddOperation(new MultiplyWeightOperation(activation1, product2));
	network.AddOperation(new ReluOperation(product2, activation2));
	network.AddOperation(new AddBiasOperation(activation2));

	network.AddOperation(new MultiplyWeightOperation(activation2, product3));
	network.AddOperation(new ReluOperation(product3, activation3));
	network.AddOperation(new AddBiasOperation(activation3));

	network.AddOperation(new MultiplyWeightOperation(activation3, product4));
	network.AddOperation(new ReluOperation(product4, activation4));
	network.AddOperation(new AddBiasOperation(activation4));
	
	network.AddOperation(new MultiplyWeightOperation(activation4, output));

	float errorSum = 0;
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
			{
				output->backwardTensor[i] = ((c >> i) & 1) - output->forwardTensor[i];
				errorSum += abs(output->backwardTensor[i]);
			}

			network.Backward();
		}
		network.Update(&UPDATE_RATE);
		
		if ((episode + 1) % EPISODES_PER_PRINT == 0)
		{
			printf("%f\n", errorSum / (BATCH_SIZE * 8 * EPISODES_PER_PRINT));
			errorSum = 0;
		}
	}

	/*network.PrintParam();
	printf("\n");

	network.PrintForward();
	printf("\n");

	network.PrintBackward();*/

	return 0;
}