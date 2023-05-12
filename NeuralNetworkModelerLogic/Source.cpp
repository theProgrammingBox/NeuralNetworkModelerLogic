#include <iostream>
#include <vector>
#include <assert.h>

/*
TOTO:
- Add in gradient arrs
- Add more concat
- Add Attention (no mask)
*/

struct Operation
{
	uint32_t outputSize;
	float* outputArr;
	Operation* inputOperation;

	virtual ~Operation() = default;
	virtual void Forward() = 0;
	virtual void Backward() = 0;
};

struct Input : Operation
{
	Input(uint32_t size)
	{
		assert(size > 0);

		this->outputSize = size;
		this->outputArr = new float[size];
		inputOperation = nullptr;
	}

	void Forward() override
	{
		printf("Input::Forward\n");
	}

	void Backward() override
	{
		printf("Input::Backward\n");
	}
};

struct Linear : Operation
{
	Linear(Operation* inputOperation, uint32_t outputSize)
	{
		assert(inputOperation != nullptr);
		assert(outputSize > 0);

		this->outputSize = outputSize;
		this->outputArr = new float[outputSize];
		inputOperation = inputOperation;
	}

	void Forward() override
	{
		printf("Linear::Forward\n");
	}

	void Backward() override
	{
		printf("Linear::Backward\n");
	}
};

struct Param3D
{
	int channels;
	int height;
	int width;
};

struct Param2D
{
	int height;
	int width;
};

struct Convolution : Operation
{
	Param3D inputParam;
	Param3D outputParam;
	Param2D kernelParam;
	Param2D paddingParam;
	Param2D strideParam;
	Param2D dilationParam;

	Convolution(Operation* inputOperation, Param3D inputParam, Param3D outputParam, Param2D kernelParam, Param2D paddingParam, Param2D strideParam, Param2D dilationParam)
	{
		assert(inputOperation != nullptr);
		assert(inputParam.channels > 0 && inputParam.height > 0 && inputParam.width > 0);
		assert(outputParam.channels > 0 && outputParam.height > 0 && outputParam.width > 0);
		assert(inputOperation->outputSize == inputParam.channels * inputParam.height * inputParam.width);
		assert(kernelParam.height > 0 && kernelParam.width > 0);
		assert(paddingParam.height >= 0 && paddingParam.width >= 0);
		assert(strideParam.height > 0 && strideParam.width > 0);
		assert(dilationParam.height >= 0 && dilationParam.width >= 0);
		assert((inputParam.height + 2 * paddingParam.height - dilationParam.height * (kernelParam.height - 1) - 1) % strideParam.height == 0);
		assert((inputParam.width + 2 * paddingParam.width - dilationParam.width * (kernelParam.width - 1) - 1) % strideParam.width == 0);

		this->inputParam = inputParam;
		this->outputParam = outputParam;
		this->kernelParam = kernelParam;
		this->paddingParam = paddingParam;
		this->strideParam = strideParam;
		this->dilationParam = dilationParam;

		outputSize = outputParam.channels * outputParam.height * outputParam.width;
		this->outputArr = new float[outputSize];
		inputOperation = inputOperation;
	}

	void Forward() override
	{
		printf("Convolution::Forward\n");
	}

	void Backward() override
	{
		printf("Convolution::Backward\n");
	}
};

struct ReLU : Operation
{
	ReLU(Operation* inputOperation)
	{
		assert(inputOperation != nullptr);
		this->outputSize = inputOperation->outputSize;
		this->outputArr = new float[outputSize];
		inputOperation = inputOperation;
	}

	void Forward() override
	{
		printf("ReLU::Forward\n");
	}

	void Backward() override
	{
		printf("ReLU::Backward\n");
	}
};

struct NeuralNetwork
{
	std::vector<Operation*> inputs;
	std::vector<Operation*> operations;
	std::vector<Operation*> outputs;

	Operation* AddInput(Operation* operation)
	{
		inputs.emplace_back(operation);
		return operation;
	}

	Operation* AddOperation(Operation* operation)
	{
		operations.emplace_back(operation);
		return operation;
	}

	Operation* AddOutput(Operation* operation)
	{
		outputs.emplace_back(operation);
		operations.emplace_back(operation);
		return operation;
	}

	void ProvideInput(uint32_t idx, float* input)
	{
		assert(idx < inputs.size());
		assert(input != nullptr);
		memcpy(inputs[idx]->outputArr, input, inputs[idx]->outputSize * sizeof(float));
	}

	void ObtainOutput(uint32_t idx, float* output)
	{
		assert(idx < outputs.size());
		assert(output != nullptr);
		memcpy(output, outputs[idx]->outputArr, outputs[idx]->outputSize * sizeof(float));
	}

	void Forward()
	{
		for (auto operation : operations)
			operation->Forward();
		printf("\n");
	}

	void Backward()
	{
		for (auto operation = operations.rbegin(); operation != operations.rend(); ++operation)
			(*operation)->Backward();
		printf("\n");
	}
};

int main()
{
	NeuralNetwork nn;
	auto input1 = nn.AddInput(new Input(64 * 64));
	auto conv1 = nn.AddOperation(new Convolution(input1, { 1, 64, 64 }, { 1, 32, 32 }, { 2, 2 }, { 0, 0 }, { 2, 2 }, { 1, 1 }));
	auto relu1 = nn.AddOperation(new ReLU(conv1));
	auto conv2 = nn.AddOperation(new Convolution(conv1, { 1, 32, 32 }, { 1, 8, 8 }, { 4, 4 }, { 0, 0 }, { 4, 4 }, { 1, 1 }));
	auto relu2 = nn.AddOperation(new ReLU(conv2));
	auto hidden1 = nn.AddOperation(new Linear(conv2, 64));
	auto relu3 = nn.AddOperation(new ReLU(hidden1));
	auto hidden2 = nn.AddOperation(new Linear(hidden1, 64));
	auto relu4 = nn.AddOperation(new ReLU(hidden2));
	auto output1 = nn.AddOutput(new Linear(hidden2, 2));

	float input[64 * 64];
	float output[2];

	nn.ProvideInput(0, input);
	nn.Forward();
	nn.ObtainOutput(0, output);
	nn.Backward();

    return 0;
}