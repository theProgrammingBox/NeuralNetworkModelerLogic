#include <iostream>
#include <vector>
#include <assert.h>

/*
Brainstorm:
- the modeler should be a sort of directed graph of gpu pointers and operations
- deep insances of these models are needed for storage of internal states and gradients
- concate operator requires space in memory that is ajacent to each other (or do a copy)
- don't deal with dimentions, just pass in pointers to gpu arrs. assert that the
flattened dimentions are correct first
*/

struct Layer
{
	int size;

	Layer(int size)
	{
		assert(size > 0);
		this->size = size;
	}
};

class Operation
{
public:
	Layer* inputLayer;
	Layer* outputLayer;

	Layer* getOutput()
	{
		return outputLayer;
	}
};

class Linear : public Operation
{
public:
	Linear(Layer* input, int outputSize)
	{
		inputLayer = input;
		outputLayer = new Layer(outputSize);
	}
};

struct Size3D
{
	int channels;
	int height;
	int width;
};

struct Size2D
{
	int height;
	int width;
};

class Convolution : public Operation
{
public:
	Convolution(Layer* input, Size3D inputSize, Size3D outputSize, Size2D kernel, Size2D padding, Size2D stride, Size2D dilation)
	{
		assert(inputSize.channels > 0 && inputSize.height > 0 && inputSize.width > 0);
		assert(outputSize.channels > 0 && outputSize.height > 0 && outputSize.width > 0);
		assert(input->size == inputSize.channels * inputSize.height * inputSize.width);
		assert(kernel.height > 0 && kernel.width > 0);
		assert(padding.height >= 0 && padding.width >= 0);
		assert(stride.height > 0 && stride.width > 0);
		assert(dilation.height >= 0 && dilation.width >= 0);
		assert((inputSize.height + 2 * padding.height - dilation.height * (kernel.height - 1) - 1) % stride.height == 0);
		assert((inputSize.width + 2 * padding.width - dilation.width * (kernel.width - 1) - 1) % stride.width == 0);
		
		inputLayer = input;
		outputLayer = new Layer(outputSize.channels * outputSize.height * outputSize.width);

		// kernel channels = input channels
		// kernel num = output channels
	}
};

class ReLU : public Operation
{
public:
	ReLU(Layer* input)
	{
		inputLayer = input;
		outputLayer = new Layer(input->size);
	}
};

struct ModelModeler
{
	std::vector<Layer*> startingLayers;
	std::vector<Layer*> middleLayers;
	std::vector<Layer*> endLayers;

	Layer* expect(Layer* layer)
	{
		startingLayers.push_back(layer);
		return layer;
	}

	Layer* add(Operation* op)
	{
		middleLayers.push_back(op->getOutput());
		return op->getOutput();
	}

	Layer* deliver(Operation* op)
	{
		endLayers.push_back(op->getOutput());
		return op->getOutput();
	}

	void compile()
	{
		
	}
};

int main()
{
	ModelModeler modeler;
	Layer* input = modeler.expect(new Layer(64 * 64));
	Layer* conv1 = modeler.add(new Convolution(input, { 1, 64, 64 }, { 1, 32, 32 }, { 2, 2 }, { 0, 0 }, { 2, 2 }, { 1, 1 }));
	Layer* relu1 = modeler.add(new ReLU(conv1));
	Layer* conv2 = modeler.add(new Convolution(conv1, { 1, 32, 32 }, { 1, 8, 8 }, { 4, 4 }, { 0, 0 }, { 4, 4 }, { 1, 1 }));
	Layer* relu2 = modeler.add(new ReLU(conv2));
	Layer* hidden1 = modeler.add(new Linear(conv2, 64));
	Layer* relu3 = modeler.add(new ReLU(hidden1));
	Layer* hidden2 = modeler.add(new Linear(hidden1, 64));
	Layer* relu4 = modeler.add(new ReLU(hidden2));
	Layer* output = modeler.deliver(new Linear(hidden2, 2));
	modeler.compile();

    return 0;
}