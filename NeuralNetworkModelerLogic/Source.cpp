#include "AddOperation.h"
#include "ReluOperation.h"
#include "GeluOperation.h"
#include "AddBiasOperation.h"
#include "MultiplyWeightOperation.h"
#include "LayerNormOperation.h"
#include "SigmoidOperation.h"
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
- implement relative rl
	- change everything from a sum approach to a set approach
		- redesign with a more "compiled" approach for a more user friendly experience
	- add concat
	- seperate action from hidden mem for cleaner code
	- add batch into the height of the matrix for speed up
- add adam optimizer
- test other architectures (like residual using layer norm)
- try other initialization methods
*/

int main()
{
	srand(time(NULL));
	
	const float LEARNING_RATE = 0.01f;
	const int BATCH_SIZE = 1;
	const int EPISODES = 1;
	const int LOG_LENGTH = 1;
	const int EPISODES_PER_PRINT = EPISODES / LOG_LENGTH;
	const int PLAYERS = 2;
	const int ACTIONS = 3;
	
	float UPDATE_RATE = LEARNING_RATE * InvSqrt(BATCH_SIZE);

	NeuralNetwork policyNetwork;

	TensorNode* policyInput = policyNetwork.AddTensorNode(new TensorNode("policyInput", 2, PLAYERS));

	TensorNode* policyProduct1 = policyNetwork.AddTensorNode(new TensorNode("policyProduct1", 4, PLAYERS));
	TensorNode* policyActivation1 = policyNetwork.AddTensorNode(new TensorNode("policyActivation1", 4, PLAYERS));
	
	TensorNode* policyOutputProduct = policyNetwork.AddTensorNode(new TensorNode("policyOutputProduct", 4, PLAYERS));
	TensorNode* policyOutputActivation = policyNetwork.AddTensorNode(new TensorNode("policyOutputActivation", 4, PLAYERS));

	policyNetwork.AddOperation(new MultiplyWeightOperation(policyInput, policyProduct1));
	policyNetwork.AddOperation(new AddBiasOperation(policyProduct1));
	policyNetwork.AddOperation(new ReluOperation(policyProduct1, policyActivation1));
	
	policyNetwork.AddOperation(new AddBiasOperation(policyActivation1));
	policyNetwork.AddOperation(new MultiplyWeightOperation(policyActivation1, policyOutputProduct));
	policyNetwork.AddOperation(new ReluOperation(policyOutputProduct, policyOutputActivation));

	
	NeuralNetwork valueNetwork;

	TensorNode* valueProduct1 = valueNetwork.AddTensorNode(new TensorNode("valueProduct1", 8, PLAYERS));
	TensorNode* valueActivation1 = valueNetwork.AddTensorNode(new TensorNode("valueActivation1", 8, PLAYERS));

	TensorNode* valueProduct2 = valueNetwork.AddTensorNode(new TensorNode("valueProduct2", 8, PLAYERS));
	TensorNode* valueActivation2 = valueNetwork.AddTensorNode(new TensorNode("valueActivation2", 8, PLAYERS));

	TensorNode* valueProduct3 = valueNetwork.AddTensorNode(new TensorNode("valueProduct3", 8, PLAYERS));
	TensorNode* valueActivation3 = valueNetwork.AddTensorNode(new TensorNode("valueActivation3", 8, PLAYERS));

	TensorNode* valueOutputProduct = valueNetwork.AddTensorNode(new TensorNode("valueOutputProduct", 1, PLAYERS));
	TensorNode* valueOutputActivation = valueNetwork.AddTensorNode(new TensorNode("valueOutputActivation", 1, PLAYERS));

	valueNetwork.AddOperation(new MultiplyWeightOperation(policyOutputActivation, valueProduct1));
	valueNetwork.AddOperation(new AddBiasOperation(valueProduct1));
	valueNetwork.AddOperation(new ReluOperation(valueProduct1, valueActivation1));
	
	valueNetwork.AddOperation(new AddBiasOperation(valueActivation1));
	valueNetwork.AddOperation(new MultiplyWeightOperation(valueActivation1, valueProduct2));
	valueNetwork.AddOperation(new ReluOperation(valueProduct2, valueActivation2));
	
	valueNetwork.AddOperation(new AddBiasOperation(valueActivation2));
	valueNetwork.AddOperation(new MultiplyWeightOperation(valueActivation2, valueProduct3));
	valueNetwork.AddOperation(new ReluOperation(valueProduct3, valueActivation3));
	
	valueNetwork.AddOperation(new AddBiasOperation(valueActivation3));
	valueNetwork.AddOperation(new MultiplyWeightOperation(valueActivation3, valueOutputProduct));
	valueNetwork.AddOperation(new SigmoidOperation(valueOutputProduct, valueOutputActivation));
	
	
	float errorSum = 0;
	for (int episode = 0; episode < EPISODES; episode++)
	{
		for (int batch = 0; batch < BATCH_SIZE; batch++)
		{
			policyNetwork.ZeroForward();
			valueNetwork.ZeroForward();
			
			for (int i = 0; i < policyInput->size; i++)
				policyInput->forwardTensor[i] = RandomFloat();

			policyNetwork.Forward();
			valueNetwork.Forward();

			int width = policyOutputActivation->width;
			int actions[PLAYERS];
			for (int player = 0; player < PLAYERS; player++)
			{
				float action = 0;
				float max = policyOutputActivation->forwardTensor[player * width];
				for (int i = 1; i < ACTIONS; i++)
				{
					if (policyOutputActivation->forwardTensor[player * width + i] > max)
					{
						max = policyOutputActivation->forwardTensor[player * width + i];
						action = i;
					}
				}
				actions[player] = action;
			}

			// print actions
			for (int player = 0; player < PLAYERS; player++)
				printf("%d ", actions[player]);
			printf("\n\n");

			/*network.ZeroBackward();
			for (int i = 0; i < 8; i++)
			{
				output->backwardTensor[i] = ((c >> i) & 1) - output->forwardTensor[i];
				errorSum += abs(output->backwardTensor[i]);
			}

			network.Backward();*/
		}
		//network.Update(&UPDATE_RATE);
		
		/*if ((episode + 1) % EPISODES_PER_PRINT == 0)
		{
			printf("%f\n", errorSum / (BATCH_SIZE * 8 * EPISODES_PER_PRINT));
			errorSum = 0;
		}*/
	}
	
	/*policyNetwork.PrintParam();
	printf("\n");
	
	valueNetwork.PrintParam();
	printf("\n");*/
	
	
	policyNetwork.PrintForward();
	printf("\n");
	
	valueNetwork.PrintForward();
	printf("\n");
	

	/*policyNetwork.PrintBackward();
	printf("\n");
	
	valueNetwork.PrintBackward();*/

	return 0;
}