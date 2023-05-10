#include <iostream>
#include <vector>

struct environment
{
	void reset(float* observation)
	{
		//
	}
};

struct ModelModeler
{
	void add()
	{
		//
	}
};

int main()
{
	environment env;
	std::vector<float*> observation;

	ModelModeler modeler;
	float* input = modeler.expect(10);
	float* hidden1 = modeler.add(input, 64);
	float* relu1 = modeler.add(hidden1, relu);
	float* hidden2 = modeler.add(relu1, 64);
	float* relu2 = modeler.add(hidden2, relu);
	float* output = modeler.give(relu2, 2);

	float* obs = new float[10];
	env.reset(obs);

    return 0;
}
