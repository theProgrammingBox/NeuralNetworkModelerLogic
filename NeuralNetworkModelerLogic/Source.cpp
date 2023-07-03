#include <iostream>
#include <vector>

/*
TODO:
- maybe insead of sum based, make sum an exclusive operation
*/

const float ONEF = 1.0f;
const float ZEROF = 0.0f;

void cpuSaxpy(
	int n,
	const float* alpha,
	const float* x, int incx,
	float* y, int incy)
{
	for (int i = 0; i < n; i++)
		y[i * incy] = *alpha * x[i * incx] + y[i * incy];
}

void PrintMatrixf32(float* arr, uint32_t rows, uint32_t cols, const char* label)
{
	printf("%s:\n", label);
	for (uint32_t i = 0; i < rows; i++)
	{
		for (uint32_t j = 0; j < cols; j++)
			printf("%8.3f ", arr[i * cols + j]);
		printf("\n");
	}
	printf("\n");
}

void cpuReluForward(
	int n,
	const float* alpha,
	const float* x,
	const float* beta,
	float* y)
{
	for (int i = 0; i < n; i++)
		y[i] = *beta * y[i] + (*alpha * x[i] >= 0 ? *alpha * x[i] : 0);
}

void cpuReluBackward(
	int n,
	const float* alpha,
	const float* dy,
	const float* x,
	const float* beta,
	float* dx)
{
	for (int i = 0; i < n; i++)
		dx[i] = *beta * dx[i] + (*alpha * x[i] >= 0 ? *alpha * dy[i] : 0);
}

struct TensorNode
{
	int size;
	float* forwardTensor;
	float* backwardTensor;

	TensorNode(int size)
		: size(size)
	{
		forwardTensor = new float[size];
		backwardTensor = new float[size];
	}

	~TensorNode()
	{
		delete[] forwardTensor;
		delete[] backwardTensor;
	}
};

struct Operation
{
	virtual ~Operation() = default;
	virtual void Forward() = 0;
};

struct AddOperation : Operation
{
	int size;
	const float* alpha;
	int incx;
	int incy;
	TensorNode* input1;
	TensorNode* input2;

	AddOperation(TensorNode* input1, TensorNode* input2, int size = 0, const float* alpha = &ONEF, int incx = 1, int incy = 1)
		: input1(input1), input2(input2), size(size), alpha(alpha), incx(incx), incy(incy)
	{
		if (size == 0)
			this->size = input1->size;
	}

	void Forward() override
	{
		cpuSaxpy(size, alpha, input1->forwardTensor, incx, input2->forwardTensor, incy);
	}
};

struct ReluOperation : Operation
{
	int size;
	const float* alpha;
	const float* beta;
	TensorNode* input;
	TensorNode* output;

	ReluOperation(TensorNode* input, TensorNode* output = nullptr, int size = 0, const float* alpha = &ONEF, const float* beta = &ZEROF)
		: input(input), output(output), size(size), alpha(alpha), beta(beta)
	{
		if (size == 0)
			this->size = input->size;
		if (output == nullptr)
			this->output = input;
	}

	void Forward() override
	{
		cpuReluForward(size, alpha, input->forwardTensor, beta, output->forwardTensor);
	}
};

struct AddBiasOperation : Operation
{
	int size;
	const float* alpha;
	int incx;
	int incy;
	TensorNode* input;
	float* bias;

	AddBiasOperation(TensorNode* input, int size = 0, const float* alpha = &ONEF, int incx = 1, int incy = 1)
		: input(input), size(size), alpha(alpha), incx(incx), incy(incy)
	{
		if (size == 0)
			this->size = input->size;
		bias = new float[input->size];
		//memset(bias, 0, sizeof(float) * input->size);
		for (int i = 0; i < input->size; i++)
			bias[i] = 1.0f;
	}

	~AddBiasOperation()
	{
		delete[] bias;
	}

	void Forward() override
	{
		cpuSaxpy(size, alpha, bias, incx, input->forwardTensor, incy);
	}
};

struct NeuralNetwork
{
	std::vector<TensorNode*> nodes;
	std::vector<Operation*> operations;

	~NeuralNetwork()
	{
		for (TensorNode* node : nodes)
			delete node;
		for (Operation* operation : operations)
			delete operation;
	}

	TensorNode* AddTensorNode(int size)
	{
		TensorNode* node = new TensorNode(size);
		nodes.emplace_back(node);
		return node;
	}

	Operation* AddOperation(Operation* operation)
	{
		operations.emplace_back(operation);
		return operation;
	}

	void Forward()
	{
		for (Operation* operation : operations)
			operation->Forward();
	}
};

int main()
{
	NeuralNetwork network;
	
	TensorNode* node1 = new TensorNode(6);
	TensorNode* node2 = network.AddTensorNode(6);

	network.AddOperation(new ReluOperation(node1, node2));
	network.AddOperation(new AddOperation(node1, node2));
	network.AddOperation(new AddBiasOperation(node2));

	for (int i = 0; i < 6; i++)
		node1->forwardTensor[i] = i - 3;
	
	network.Forward();
	PrintMatrixf32(node1->forwardTensor, 1, 6, "input");
	PrintMatrixf32(node2->forwardTensor, 1, 6, "relu and sum and bias");
	
	return 0;
}