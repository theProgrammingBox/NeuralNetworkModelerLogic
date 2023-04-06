#include <cublas_v2.h>
#include <cudnn.h>
#include <curand.h>
#include <cuda_fp16.h>

#include <iostream>
#include <vector>

class Tensor
{
public:
};

class Layer
{
public:
	//Layer(Tensor* inputTensor);
	//Layer(Layer* inputLayer);
	virtual ~Layer() = default;

	Tensor* GetOutputTensor() const;
	//virtual void Forward() = 0;

private:
	Tensor* outputTensor;
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
	
private:
	Tensor* inputTensor;
	static const cudnnActivationDescriptor_t reluActivationDescriptor;
};

class SoftmaxLayer : public Layer
{
public:
	SoftmaxLayer(Tensor* inputTensor);
	SoftmaxLayer(Layer* inputLayer);
};

class SoftmaxSamplerLayer : public Layer
{
public:
	SoftmaxSamplerLayer(Tensor* inputTensor);
	SoftmaxSamplerLayer(Layer* inputLayer);
};

class NetworkModeler
{
public:
	Tensor* AddTensor();
	Layer* AddLayer(Layer* layer);
	Tensor* GetTensor(Layer* layer);
	void Forward();

private:
	static const cublasHandle_t cublasHandle;
	static const cudnnHandle_t cudnnHandle;
	static const curandGenerator_t curandGenerator;
};

int main()
{
    NetworkModeler modeler;
	
    auto inputTensor = modeler.AddTensor();
	auto layer1MatMul = modeler.AddLayer(new MatMulLayer(inputTensor));
	auto layer1MatAdd = modeler.AddLayer(new MatAddLayer(layer1MatMul));
	auto layer1Relu = modeler.AddLayer(new ReluLayer(layer1MatAdd));
	auto layer2MatMul = modeler.AddLayer(new MatMulLayer(layer1Relu));
	auto layer2MatAdd = modeler.AddLayer(new MatAddLayer(layer2MatMul));
	auto layer2Softmax = modeler.AddLayer(new SoftmaxLayer(layer2MatAdd));
	auto softmaxSampler = modeler.AddLayer(new SoftmaxSamplerLayer(layer2Softmax));
	auto outputTensor = modeler.GetTensor(softmaxSampler);

    return 0;
}
