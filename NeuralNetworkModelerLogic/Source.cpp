#include <iostream>
#include <vector>

const float ONEF = 1.0f;

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
	TensorNode* input1;
	TensorNode* input2;

	AddOperation(int size, TensorNode* input1, TensorNode* input2)
		: size(size), input1(input1), input2(input2)
	{
	}

	void Forward() override
	{
		cpuSaxpy(size, &ONEF, input1->forwardTensor, 1, input2->forwardTensor, 1);
	}
};

struct ReluOperation : Operation
{
	int size;
	TensorNode* input;
	TensorNode* output;

	ReluOperation(int size, TensorNode* input, TensorNode* output)
		: size(size), input(input), output(output)
	{
	}

	void Forward() override
	{
		cpuReluForward(size, &ONEF, input->forwardTensor, &ONEF, output->forwardTensor);
	}
};

struct AddBiasOperation : Operation
{
	TensorNode* input;
	TensorNode* output;
	float* bias;

	AddBiasOperation(TensorNode* input)
		: input(input)
	{
		bias = new float[input->size];
		for (int i = 0; i < input->size; i++)
			bias[i] = 1 - i;
	}

	~AddBiasOperation()
	{
		delete[] bias;
	}

	void Forward() override
	{
		cpuSaxpy(size, &ONEF, bias, 1, input->forwardTensor, 1);
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
		nodes.push_back(node);
		return node;
	}

	Operation* AddOperation(Operation* operation)
	{
		operations.push_back(operation);
		return operation;
	}

	void Forward()
	{
		for (TensorNode* node : nodes)
			memset(node->forwardTensor, 0, sizeof(float) * node->size);
		for (Operation* operation : operations)
			operation->Forward();
	}
};

int main()
{
	NeuralNetwork network;
	
	TensorNode* node1 = new TensorNode(6);
	TensorNode* node2 = network.AddTensorNode(6);
	TensorNode* node3 = network.AddTensorNode(6);

	network.AddOperation(new ReluOperation(6, node1, node2));
	network.AddOperation(new AddOperation(6, node1, node3));
	network.AddOperation(new AddOperation(6, node2, node3));
	network.AddOperation(new AddBiasOperation(node3));

	for (int i = 0; i < 6; i++)
		node1->forwardTensor[i] = i - 3;
	
	network.Forward();
	PrintMatrixf32(node2->forwardTensor, 1, 6, "relu");
	PrintMatrixf32(node3->forwardTensor, 1, 6, "relu and sum and bias");
	
	return 0;
}