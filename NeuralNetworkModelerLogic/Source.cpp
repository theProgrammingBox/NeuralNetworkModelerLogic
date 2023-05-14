#include <iostream>
#include <vector>
#include <assert.h>

//#include <cudnn.h>
//#include <cublas_v2.h>
//#include <curand.h>

/*
TOTO:
- rework data layout as presented in Thought Organization
(add data layouts like transpose and NCHW)
- implement cudnn and cublas operations
- implement curand random initialization (orthogonal)
- unit test cuda operations
(concerned about data types and convolution layouts)

- Add in gradient arrs
- unit test backpropagation

- Add concat
(should I just copy both inputs into a new arr?)
- Add Attention (no mask)
(rethink, transpose operations should be defined by expected inputs?)
*/

/*
Thought Organization:
- ParameterInput is a matrix that is filled by noone
(it is randomly initialized during network creation and is updated during training)
- UserInput is a matrix that the user is expected to fill before each iteration
- NetworkInput is a matrix that the network of the previous iteration is expected to fill
(generates an initial ParameterInput for the first iteration, then uses the output of the previous
iteration for the next iterations)

- so weights, biases, and initial hidden states are all ParameterInputs
- user input is UserInput
- output of the previous iteration is NetworkInput
(define a method to get all network outputs for the previous iteration to be used as NetworkInput for
the next iteration)

- I need a more detailed description for the matrixes. Use same old method of just using the total
size of the matrix, not dimentions to make it easier to work with. Basically, instead of saying
reshape to this, you just say what shape you expect in the operation. Data layout is always
the same, the operation changes.
(methodology: let user have ultimate control, but provide very basic error checking)

- include NCHW, NHWC, and transpose?
(NCHW is pixels of the width, H times, C times, N times)

- include data types? including expected operation output data types?
(cudnn and cublas types like CUDNN_DATA_FLOAT, CUDNN_DATA_HALF, CUDA_R_32F, CUDA_R_16F)
*/

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

	MatrixMultiplication(Param4D* inputParam1, Param4D* inputParam2)
	{
		assert(inputParam1 != nullptr);
		assert(inputParam2 != nullptr);
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
	Param2D kernelParam;
	Param2D paddingParam;
	Param2D strideParam;
	Param2D dilationParam;

	Convolution(Param4D* inputParam, Param4D outputParam, Param2D kernelParam, Param2D paddingParam, Param2D strideParam, Param2D dilationParam)
	{
		assert(inputParam != nullptr);
		assert(inputParam->batches == outputParam.batches);
		assert(kernelParam.height > 0 && kernelParam.width > 0);
		assert(strideParam.height > 0 && strideParam.width > 0);
		assert(float(inputParam->height + 2 * paddingParam.height - dilationParam.height * (kernelParam.height - 1) - 1) / strideParam.height + 1 == outputParam.height);
		assert(float(inputParam->width + 2 * paddingParam.width - dilationParam.width * (kernelParam.width - 1) - 1) / strideParam.width + 1 == outputParam.width);

		this->inputParam = inputParam;
		this->outputParam = new Param4D(outputParam);
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

struct NeuralNetwork
{
	std::vector<Param4D*> parameters;
	std::vector<Param4D*> userInputs;
	std::vector<Param4D*> networkInputs;
	std::vector<Param4D*> userOutputs;
	std::vector<Param4D*> networkOutputs;

	std::vector<Operation*> operations;

	~NeuralNetwork()
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
	NeuralNetwork nn;

	auto input1 = nn.UserInput(Param4D(64, 64));
	auto conv1 = nn.AddOperation(Convolution(input1, Param4D(32, 32), Param2D(2, 2), Param2D(0, 0), Param2D(2, 2), Param2D(1, 1)));
	auto relu1 = nn.AddOperation(ReLU(conv1));

	auto conv2 = nn.AddOperation(Convolution(conv1, Param4D(8, 8), Param2D(4, 4), Param2D(0, 0), Param2D(4, 4), Param2D(1, 1)));
	auto relu2 = nn.AddOperation(ReLU(conv2));

	auto flatten1 = relu2->Reshape(Param4D(1, 64));
	auto weight1 = nn.ParameterInput(Param4D(64, 64));
	auto hidden1 = nn.AddOperation(MatrixMultiplication(flatten1, weight1));
	auto relu3 = nn.AddOperation(ReLU(hidden1));

	auto weight2 = nn.ParameterInput(Param4D(64, 64));
	auto hidden2 = nn.AddOperation(MatrixMultiplication(hidden1, weight2));
	auto relu4 = nn.AddOperation(ReLU(hidden2));

	auto weight3 = nn.ParameterInput(Param4D(64, 2));
	auto output1 = nn.AddOperation(MatrixMultiplication(hidden2, weight3));

    return 0;
}