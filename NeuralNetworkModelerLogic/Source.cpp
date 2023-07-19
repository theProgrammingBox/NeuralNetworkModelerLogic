#include "AddOperation.h"
#include "ReluOperation.h"
#include "GeluOperation.h"
#include "AddBiasOperation.h"
#include "MultiplyWeightOperation.h"
#include "LayerNormOperation.h"
#include "NeuralNetwork.h"

/*
UNCERTAIN BUGS:
- random nans sometimes when near the end
*/

/*
IMPORTANT LESSONS:
- increasing batch size is a good way to "speed" up training when increasing or decreasing learning rate doesn't work
- where you place your biases matters, there are dynamics that are not to intuitive going on during backprop
- layer norm does not replace an activation function, it is a normalization technique
- the scale and purpose of the model decides the usefulness of different operations
*/

/*
TODO:
- add adam optimizer
- test other architectures (like residual using layer norm)
- try other initialization methods
*/

int main()
{
	srand(time(NULL));
	
	const float LEARNING_RATE = 0.006f;
	const int BATCH_SIZE = 64;
	const int EPISODES = 1000000;
	const int LOG_LENGTH = 128;
	const int EPISODES_PER_PRINT = EPISODES / LOG_LENGTH;
	
	float UPDATE_RATE = LEARNING_RATE * InvSqrt(BATCH_SIZE);

	NeuralNetwork network;

	TensorNode* input = network.AddTensorNode(new TensorNode("input", 16));
	
	TensorNode* product1 = network.AddTensorNode(new TensorNode("product1", 16));
	TensorNode* activation1 = network.AddTensorNode(new TensorNode("activation1", 16));
	
	TensorNode* product2 = network.AddTensorNode(new TensorNode("product2", 16));
	TensorNode* activation2 = network.AddTensorNode(new TensorNode("activation2", 16));

	TensorNode* product3 = network.AddTensorNode(new TensorNode("product3", 16));
	TensorNode* activation3 = network.AddTensorNode(new TensorNode("activation3", 16));

	TensorNode* product4 = network.AddTensorNode(new TensorNode("product3", 16));
	TensorNode* activation4 = network.AddTensorNode(new TensorNode("activation3", 16));

	TensorNode* product5 = network.AddTensorNode(new TensorNode("product3", 16));
	TensorNode* activation5 = network.AddTensorNode(new TensorNode("activation3", 16));

	TensorNode* output = network.AddTensorNode(new TensorNode("output", 8));

	network.AddOperation(new MultiplyWeightOperation(input, product1));
	network.AddOperation(new AddBiasOperation(product1));
	network.AddOperation(new GeluOperation(product1, activation1));
	network.AddOperation(new AddBiasOperation(activation1));
	
	network.AddOperation(new MultiplyWeightOperation(activation1, product2));
	network.AddOperation(new GeluOperation(product2, activation2));
	network.AddOperation(new AddBiasOperation(activation2));
	
	network.AddOperation(new MultiplyWeightOperation(activation2, product3));
	network.AddOperation(new GeluOperation(product3, activation3));
	network.AddOperation(new AddBiasOperation(activation3));

	network.AddOperation(new MultiplyWeightOperation(activation3, product4));
	network.AddOperation(new GeluOperation(product4, activation4));
	network.AddOperation(new AddBiasOperation(activation4));

	network.AddOperation(new MultiplyWeightOperation(activation4, product5));
	network.AddOperation(new GeluOperation(product5, activation5));
	network.AddOperation(new AddBiasOperation(activation5));
	
	network.AddOperation(new MultiplyWeightOperation(activation5, output));

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

	network.PrintParam();
	printf("\n");

	network.PrintForward();
	printf("\n");

	network.PrintBackward();/**/

	return 0;
}