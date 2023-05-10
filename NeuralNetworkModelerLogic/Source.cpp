#include <iostream>
#include <vector>
#include <assert.h>

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

struct Size {
	int channels;
	int height;
	int width;
};

struct Kernel {
	int num_kernels;
	int height;
	int width;
};

struct Padding {
	int height;
	int width;
};

struct Stride {
	int height;
	int width;
};

struct Dilation {
	int height;
	int width;
};

class Convolution : public Operation
{
public:
	// requirments: kernel size, padding, stride, dilation, input size(channel, height, width), output size(channel, height, width)
	Convolution(Layer* input, Size inputSize, Size outputSize, Kernel kernel, Padding padding, Stride stride, Dilation dilation)
	{
		assert(input->size == inputSize.channels * inputSize.height * inputSize.width);
		assert(kernel.num_kernels == outputSize.channels);
		assert(kernel.height > 0 && kernel.width > 0);
		assert(padding.height >= 0 && padding.width >= 0);
		assert(stride.height > 0 && stride.width > 0);
		assert(dilation.height >= 0 && dilation.width >= 0);
		assert((inputSize.height + 2 * padding.height - dilation.height * (kernel.height - 1) - 1) % stride.height == 0);
		assert((inputSize.width + 2 * padding.width - dilation.width * (kernel.width - 1) - 1) % stride.width == 0);
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
		/*op.generateLayers(&layers);
		if (act != nullptr)
		{
			act.generateLayers(&layers);
			return act.getOutputLayer();
		}
		return op.getOutputLayer();*/
	}
};

int main()
{
	ModelModeler modeler;
	Layer* input = modeler.expect(new Layer(4096));
	Layer* conv1 = modeler.add(new Convolution(input, 1024), new ReLU());
	Layer* conv2 = modeler.add(new Convolution(conv1, 64), new ReLU());
	Layer* hidden1 = modeler.add(new Linear(conv2, 64), new ReLU());
	Layer* hidden2 = modeler.add(new Linear(hidden1, 64), new ReLU());
	Layer* output = modeler.add(new Linear(hidden2, 2));

    return 0;
}
