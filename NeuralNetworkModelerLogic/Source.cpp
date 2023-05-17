#include <iostream>
#include <vector>
#include <assert.h>

/*
Description:
- This is a neural network modeler that allows the user to create a neural network.
- the goal is to allow the user to have complete freedom in designing every aspect of the
operations used in the neural network. Would you like the network to output its own kernel
weights for the next convolution? You can do that. Would you like mix different data
types and layouts? You can do that.
- TLDR: A project that allows the user to create a neural network at a very low level.
*/

/*
TOTO:
- allow more less detailed data layouts 
- add transpose options to matrix multiplication
- rework data layout as presented in Thought Organization
- implement cudnn and cublas operations
- implement curand random initialization (orthogonal)
- unit test cuda operations

- Add in gradient arrs
- unit test backpropagation

- work out multi head attention
- Add concat
- Add split

- Add in auto order operations (compile())
*/

/*
Thought Organization:
- use NCHW and float by default. work towards other datatypes afterwards
(NCHW is pixels of the width, H times, C times, N times)

- include data types? including expected operation output data types?
(cudnn and cublas types like CUDNN_DATA_FLOAT, CUDNN_DATA_HALF, CUDA_R_32F, CUDA_R_16F)
*/

/*
struct Param4D
{
	uint32_t height;
	uint32_t width;
	uint32_t channels;
	uint32_t batches;

	Param4D(uint32_t height = 1, uint32_t width = 1, uint32_t channels = 1, uint32_t batches = 1)
	{
		assert(height > 0 && width > 0 && channels > 0 && batches > 0);
		this->height = height;
		this->width = width;
		this->channels = channels;
		this->batches = batches;
	}

	Param4D(const Param4D* other)
	{
		this->height = other->height;
		this->width = other->width;
		this->channels = other->channels;
		this->batches = other->batches;
	}

	bool operator==(const Param4D& other) const
	{
		return height == other.height && width == other.width && channels == other.channels && batches == other.batches;
	}

	uint32_t size() const
	{
		return width * height * channels * batches;
	}

	Param4D* Reshape(Param4D param) const
	{
		assert(param.size() == size());
		Param4D* parameter = new Param4D(param);
		return parameter;
	}

	void Print() const
	{
		printf("(%u, %u, %u, %u)\n", height, width, channels, batches);
	}
};

struct Param2D
{
	uint32_t height;
	uint32_t width;

	Param2D(uint32_t height = 1, uint32_t width = 1)
	{
		assert(height >= 0 && width >= 0);
		this->height = height;
		this->width = width;
	}
};

struct Operation
{
	Param4D* outputParam;
};

struct MatrixMultiplication : Operation
{
	Param4D* inputParam1;
	Param4D* inputParam2;

	MatrixMultiplication(Param4D* inputParam1, Param4D* inputParam2, Param2D transpose = {0, 0})
	{
		assert(inputParam1 != nullptr);
		assert(inputParam2 != nullptr);
		// transpose logic asserts
		assert(inputParam1->width == inputParam2->height);
		assert(inputParam1->batches == inputParam2->batches);

		this->inputParam1 = inputParam1;
		this->inputParam2 = inputParam2;
		outputParam = new Param4D(inputParam1->height, inputParam2->width, 1, inputParam1->batches);
	}
};

struct MatrixAddition : Operation
{
	Param4D* inputParam1;
	Param4D* inputParam2;

	MatrixAddition(Param4D* inputParam1, Param4D* inputParam2)
	{
		assert(inputParam1 != nullptr);
		assert(inputParam2 != nullptr);
		assert(*inputParam1 == *inputParam2);

		this->inputParam1 = inputParam1;
		this->inputParam2 = inputParam2;
		outputParam = new Param4D(inputParam1);
	}
};

struct Convolution : Operation
{
	Param4D* inputParam;
	Param4D* kernelParam;
	Param2D strideParam;
	Param2D paddingParam;
	Param2D dilationParam;

	Convolution(Param4D* inputParam, Param4D* kernelParam, Param2D strideParam = { 1, 1 }, Param2D paddingParam = { 0, 0 }, Param2D dilationParam = { 1, 1 })
	{
		assert(inputParam != nullptr);
		assert(inputParam->channels == kernelParam->channels);
		float outputHeight = float(inputParam->height + 2 * paddingParam.height - dilationParam.height * (kernelParam->height - 1) - 1) / strideParam.height + 1;
		float outputWidth = float(inputParam->width + 2 * paddingParam.width - dilationParam.width * (kernelParam->width - 1) - 1) / strideParam.width + 1;
		assert(outputHeight == uint32_t(outputHeight));
		assert(outputWidth == uint32_t(outputWidth));

		this->inputParam = inputParam;
		this->outputParam = new Param4D(outputHeight, outputWidth, kernelParam->batches, inputParam->batches);
		this->kernelParam = kernelParam;
		this->paddingParam = paddingParam;
		this->strideParam = strideParam;
		this->dilationParam = dilationParam;
	}
};

struct ReLU : Operation
{
	Param4D* inputParam;

	ReLU(Param4D* inputParam)
	{
		assert(inputParam != nullptr);

		this->inputParam = inputParam;
		outputParam = new Param4D(inputParam);
	}
};

struct ModelModeler
{
	std::vector<Param4D*> parameters;
	std::vector<Param4D*> userInputs;
	std::vector<Param4D*> networkInputs;
	std::vector<Param4D*> userOutputs;
	std::vector<Param4D*> networkOutputs;

	std::vector<Operation*> operations;

	~ModelModeler()
	{
		for (Param4D* parameter : parameters)
			delete parameter;
		for (Param4D* parameter : userInputs)
			delete parameter;
		for (Param4D* parameter : networkInputs)
			delete parameter;
		for (Param4D* parameter : userOutputs)
			delete parameter;
		for (Param4D* parameter : networkOutputs)
			delete parameter;
		for (Operation* operation : operations)
			delete operation;
	}

	Param4D* ParameterInput(Param4D param)
	{
		Param4D* parameter = new Param4D(param);
		parameters.emplace_back(parameter);
		return parameter;
	}

	Param4D* UserInput(Param4D param)
	{
		Param4D* parameter = new Param4D(param);
		userInputs.emplace_back(parameter);
		return parameter;
	}

	Param4D* NetworkInput(Param4D param)
	{
		Param4D* parameter = new Param4D(param);
		networkInputs.emplace_back(parameter);
		return parameter;
	}

	Param4D* AddOperation(Operation op)
	{
		Operation* operation = new Operation(op);
		operations.emplace_back(operation);
		return operation->outputParam;
	}
};

int main()
{
	ModelModeler nn;

	auto input1 = nn.UserInput({ 4096 });

	auto flatten0 = input1->Reshape({ 64, 64 });
	auto kernel1 = nn.ParameterInput({ 2, 2, 1, 8 });
	auto conv1 = nn.AddOperation(Convolution(input1, kernel1, { 2, 2 }));
	auto relu1 = nn.AddOperation(ReLU(conv1));

	auto kernel2 = nn.ParameterInput({ 4, 4, kernel1->batches, 32 });
	auto conv2 = nn.AddOperation(Convolution(conv1, kernel2, { 4, 4 }));
	auto relu2 = nn.AddOperation(ReLU(conv2));

	auto flatten1 = relu2->Reshape({ 1, relu2->size()});
	auto weight1 = nn.ParameterInput({ flatten1->width, 64 });
	auto hidden1 = nn.AddOperation(MatrixMultiplication(flatten1, weight1));
	auto relu3 = nn.AddOperation(ReLU(hidden1));

	auto weight2 = nn.ParameterInput({ relu3->width, 64 });
	auto hidden2 = nn.AddOperation(MatrixMultiplication(hidden1, weight2));
	auto relu4 = nn.AddOperation(ReLU(hidden2));

	auto weight3 = nn.ParameterInput({ relu4->width, 2 });
	auto output1 = nn.AddOperation(MatrixMultiplication(hidden2, weight3));

    return 0;
}
*/

struct ParameterDetails
{
	uint32_t size;
	float* arr;

	ParameterDetails(uint32_t size) : size(size) {}
};

struct OperationDetails
{
	OperationDetails(ParameterDetails* inputParam1, ParameterDetails* inputParam2, ParameterDetails* outputParam)
	{
		this->inputParam1 = inputParam1;
		this->inputParam2 = inputParam2;
		this->outputParam = outputParam;
	}

	ParameterDetails* inputParam1;
	ParameterDetails* inputParam2;
	ParameterDetails* outputParam;
};

struct Operation
{
	float* inputParam1;
	float* inputParam2;
	float* outputParam;

	void Forward()
	{
		*outputParam = *inputParam1 + *inputParam2;
	}
};

struct NeuralNetwork
{
	std::vector<Operation> operations;
	std::vector<float*> inputs;
	std::vector<float*> outputs;

	~NeuralNetwork()
	{
		for (Operation operation : operations)
		{
			delete[] operation.inputParam1;
			delete[] operation.inputParam2;
			delete[] operation.outputParam;
		}
	}

	void Forward()
	{
		for (Operation operation : operations)
		{
			operation.Forward();
		}
	}
};

struct Modeler
{
	std::vector<ParameterDetails*> dynamicParameters;
	std::vector<ParameterDetails*> constantParameters;
	std::vector<ParameterDetails*> inputParameters;
	std::vector<ParameterDetails*> outputParameters;
	std::vector<OperationDetails> operations;

	~Modeler()
	{
		for (ParameterDetails* parameter : dynamicParameters)
			delete parameter;
		for (ParameterDetails* parameter : constantParameters)
			delete parameter;
	}

	ParameterDetails* AddDynamicParameter(ParameterDetails* parameter)
	{
		dynamicParameters.emplace_back(parameter);
		return parameter;
	}

	ParameterDetails* AddConstantParameter(ParameterDetails* parameter)
	{
		constantParameters.emplace_back(parameter);
		return parameter;
	}

	void HintInput(ParameterDetails* parameter)
	{
		inputParameters.emplace_back(parameter);
	}

	void HintOutput(ParameterDetails* parameter)
	{
		outputParameters.emplace_back(parameter);
	}

	void AddOperation(OperationDetails operation)
	{
		operations.emplace_back(operation);
	}

	void Initialize()
	{
		for (ParameterDetails* parameter : constantParameters)
		{
			parameter->arr = new float[parameter->size];
			for (uint32_t i = 0; i < parameter->size; i++)
				parameter->arr[i] = i;
		}
	}

	NeuralNetwork Instance()
	{
		for (ParameterDetails* parameter : dynamicParameters)
			parameter->arr = new float[parameter->size];
		
		NeuralNetwork nn;

		for (ParameterDetails* parameter : inputParameters)
			nn.inputs.emplace_back(parameter->arr);

		for (ParameterDetails* parameter : outputParameters)
			nn.outputs.emplace_back(parameter->arr);
		
		for (OperationDetails operation : operations)
		{
			Operation op;
			op.inputParam1 = operation.inputParam1->arr;
			op.inputParam2 = operation.inputParam2->arr;
			op.outputParam = operation.outputParam->arr;
			nn.operations.emplace_back(op);
		}
		return nn;
	}
};

// weights are constant params
// user inputs and operation outputs are dynamic params params
// dynamic params are dependent on other dynamic params and/or constant params
// constant params are not dependent on any other params
// we can use compile to determin if constant or dynamic

int main()
{
	float* arr = new float[1];
	float* a = arr;
	float* b = arr;

	delete[] a;
	delete[] b;
	/*Modeler modeler;
	
	ParameterDetails* input = modeler.AddDynamicParameter(new ParameterDetails(1));
	ParameterDetails* kernel = modeler.AddConstantParameter(new ParameterDetails(1));
	ParameterDetails* output = modeler.AddDynamicParameter(new ParameterDetails(1));

	modeler.HintInput(input);
	modeler.HintOutput(output);
	
	modeler.AddOperation(OperationDetails(input, kernel, output));
	modeler.Initialize();
	
	NeuralNetwork neuralNetwork = modeler.Instance();

	neuralNetwork.inputs[0][0] = 1;
	neuralNetwork.Forward();
	printf("%f\n", neuralNetwork.outputs[0][0]);*/
	
	return 0;
}