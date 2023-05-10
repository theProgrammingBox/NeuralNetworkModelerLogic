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
	float* data;
	Layer(int size) : size(size)
	{
		// gpu malloc
	}
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
	Linear(Layer* input, int outputSize)
	{
		assert(outputSize > 0);
		assert(input->size == outputSize);
	}
	void forward() override {}
	void backward() override {}
};

struct Size3D
{
	int channels;
	int height;
	int width;
};

struct size2D
{
	int height;
	int width;
};

class Convolution : public Operation
{
public:
	Convolution(Layer* input, Size3D inputSize, Size3D outputSize, size2D kernel, size2D padding, size2D stride, size2D dilation)
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

		// kernel channels = input channels
		// kernel num = output channels
	}
	void forward() override {}
	void backward() override {}
};;

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

	Layer* add(Operation* op, Activation* act = nullptr)
	{
		if (act)
		{
			// return activation new Layer
		}
		// return op new Layer
	}
};

int main()
{
	ModelModeler modeler;
	Layer* input = modeler.expect(new Layer(4096));
	// the convolution math below isn't correct, it's just a placeholder
	Layer* conv1 = modeler.add(new Convolution(input, { 1, 64, 64 }, { 1, 32, 32 }, { 3, 3 }, { 0, 0 }, { 1, 1 }, { 1, 1 }), new ReLU());
	Layer* conv2 = modeler.add(new Convolution(conv1, { 1, 32, 32 }, { 1, 8, 8 }, { 3, 3 }, { 0, 0 }, { 1, 1 }, { 1, 1 }), new ReLU());
	Layer* hidden1 = modeler.add(new Linear(conv2, 64), new ReLU());
	Layer* hidden2 = modeler.add(new Linear(hidden1, 64), new ReLU());
	Layer* output = modeler.add(new Linear(hidden2, 2));

    return 0;
}
