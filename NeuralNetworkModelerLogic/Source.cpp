#include <iostream>
#include <vector>

/*
Brainstorm:
- the modeler should be a sort of directed graph of gpu pointers and operations
- deep insances of these models are needed for storage of internal states and gradients
- concate operator requires space in memory that is ajacent to each other (or do a copy)
*/

struct Layer
{
	int size;
	Layer(int size) : size(size) {}
};

class Operation
{
public:
	virtual void forward() = 0;
	virtual void backward() = 0;
};

class Linear : public Operation
{
public:
	Linear(Layer* input, int outputSize) {}
	void forward() override {}
	void backward() override {}
};

class Activation {
public:
	virtual void forward() = 0;
	virtual void backward() = 0;
};

class ReLU : public Activation
{
public:
	void forward() override {}
	void backward() override {}
};

struct ModelModeler
{
	std::vector<Layer*> layers;
	Layer* expect(Layer* layer)
	{
		layers.push_back(layer);
		return layer;
	}
	Layer add(Operation* op, Activation* act) {}
};

struct Layer
{
};

int main()
{
	ModelModeler modeler;
	Layer* input = modeler.expect(new Layer(10));
	Layer* hidden1 = modeler.add(new Linear(input, 20), new ReLU());

    return 0;
}
