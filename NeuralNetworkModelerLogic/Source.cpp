#include <iostream>
#include <vector>

class Tensor
{
public:
};

class Layer
{
public:
	float* outputTensor;
};

class MatMulLayer : public Layer
{
public:
	MatMulLayer(Tensor* inputTensor);
	MatMulLayer(Layer* inputLayer);
};

class MatAddLayer : public Layer
{
public:
	MatAddLayer(Tensor* inputTensor);
	MatAddLayer(Layer* inputLayer);
};

class ReluLayer : public Layer
{
public:
	ReluLayer(Tensor* inputTensor);
	ReluLayer(Layer* inputLayer);
};

class SoftmaxLayer : public Layer
{
public:
	SoftmaxLayer(Tensor* inputTensor);
	SoftmaxLayer(Layer* inputLayer);
};

class SampleSoftmaxLayer : public Layer
{
public:
	SampleSoftmaxLayer(Tensor* inputTensor);
	SampleSoftmaxLayer(Layer* inputLayer);
};

class NetworkModeler
{
public:
	Tensor* AddInputTensor();
	Layer* AddLayer(Layer* layer);
	Tensor* GetOutputTensor(Layer* layer);
};

int main()
{
    NetworkModeler modeler;
	
    auto inputTensor = modeler.AddInputTensor();
	auto layer1MatMul = modeler.AddLayer(new MatMulLayer(inputTensor));
	auto layer1MatAdd = modeler.AddLayer(new MatAddLayer(layer1MatMul));
	auto layer1Relu = modeler.AddLayer(new ReluLayer(layer1MatAdd));
	auto layer2MatMul = modeler.AddLayer(new MatMulLayer(layer1Relu));
	auto layer2MatAdd = modeler.AddLayer(new MatAddLayer(layer2MatMul));
	auto layer2Softmax = modeler.AddLayer(new SoftmaxLayer(layer2MatAdd));
	auto sampleSoftmax = modeler.AddLayer(new SampleSoftmaxLayer(layer2Softmax));
	auto outputTensor = modeler.GetOutputTensor(sampleSoftmax);

    return 0;
}
