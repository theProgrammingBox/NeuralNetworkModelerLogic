#include "BasicResidualOperation.h"

#include "NeuralNetwork.h"

/*
Warnings:
*/

/*
TODO:
- add a compile function to allow for printing, forward, and backward
-- there is a problem where if you make integrated operations, zeroForward doesn't touch the integrated parameters
-- same problem occurs with printing and others
-- compiling will do an ordered search to make an ordered list of unique items
-- i might find a better solution then before as I was avoiding compiling
- add layer norm
*/

int main()
{
	const float LEARNING_RATE = 0.002f;
	const int BATCH_SIZE = 1;
	const int EPISODES = 10;

	float UPDATE_RATE = LEARNING_RATE / BATCH_SIZE;
	
	NeuralNetwork network;
	
	TensorNode* input = network.AddTensorNode(new TensorNode("input", 8));
	TensorNode* output = network.AddTensorNode(new TensorNode("output", 8));

	network.AddOperation(new BasicResidualOperation(input, output));

	/*for (int episode = 0; episode < EPISODES; episode++)
	{
		for (int batch = 0; batch < BATCH_SIZE; batch++)
		{
			network.ZeroForward();
			for (int i = 0; i < 8; i++)
				input->forwardTensor[i] = i;

			network.Forward();

			network.ZeroBackward();
			for (int i = 0; i < 8; i++)
				output->backwardTensor[i] = input->forwardTensor[i] - output->forwardTensor[i];

			network.Backward();
		}
		network.Update(&UPDATE_RATE);
	}*/
	
	network.PrintForward();
	printf("\n");

	network.PrintBackward();
	printf("\n");
	
	network.PrintParam();
	
	return 0;
}