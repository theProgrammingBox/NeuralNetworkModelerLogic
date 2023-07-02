#include <iostream>
#include <vector>

const float ONEF = 1.0f;
const float MINUS_ONEF = -1.0f;

/*
TODO:
*/

void cpuSgemmStridedBatched(
	bool transB, bool transA,
	int CCols, int CRows, int AColsBRows,
	const float* alpha,
	const float* B, int ColsB, int SizeB,
	const float* A, int ColsA, int SizeA,
	const float* beta,
	float* C, int ColsC, int SizeC,
	int batchCount)
{
	for (int b = batchCount; b--;)
	{
		for (int m = CCols; m--;)
			for (int n = CRows; n--;)
			{
				float sum = 0;
				for (int k = AColsBRows; k--;)
					sum += (transA ? A[k * ColsA + n] : A[n * ColsA + k]) * (transB ? B[m * ColsB + k] : B[k * ColsB + m]);
				C[n * ColsC + m] = *alpha * sum + *beta * C[n * ColsC + m];
			}
		A += SizeA;
		B += SizeB;
		C += SizeC;
	}
}

void PrintMatrixf32(float* arr, uint32_t rows, uint32_t cols, const char* label)
{
	printf("%s:\n", label);
	for (uint32_t i = 0; i < rows; i++)
	{
		for (uint32_t j = 0; j < cols; j++)
			printf("%8.4f ", arr[i * cols + j]);
		printf("\n");
	}
	printf("\n");
}

struct Operation
{
	virtual ~Operation() = default;
	virtual float* GetOutputTensor() = 0;
	virtual float* GetInputGradientTensor() = 0;
	virtual void ZeroForward() = 0;
	virtual void ZeroBackward() = 0;
	virtual void Forward() = 0;
	virtual void Backward() = 0;
	virtual void Update() = 0;
};

struct Linear : Operation
{
	int inputTensorHeight;
	int inputTensorWidth;
	int outputTensorWidth;
	float* inputTensor;
	float* weightTensor;
	float* outputTensor;
	float* outputGradientTensor;
	
	Linear(int inputTensorHeight, int inputTensorWidth, int outputTensorWidth, float* inputTensor, float* outputGradientTensor) :
		inputTensorHeight(inputTensorHeight),
		inputTensorWidth(inputTensorWidth),
		outputTensorWidth(outputTensorWidth),
		inputTensor(inputTensor),
		outputGradientTensor(outputGradientTensor)
	{
		weightTensor = new float[inputTensorWidth * outputTensorWidth];
		outputTensor = new float[inputTensorHeight * outputTensorWidth];
		memset(weightTensor, 0, sizeof(float) * inputTensorWidth * outputTensorWidth);
		for (int i = 0; i < inputTensorWidth; i++)
			weightTensor[i * outputTensorWidth + i] = 1.0f;
		//PrintMatrixf32(weightTensor, inputTensorWidth, outputTensorWidth, "weightTensor");
	}
	
	~Linear() override
	{
		delete[] weightTensor;
		delete[] outputTensor;
	}

	float* GetOutputTensor() override
	{
		return outputTensor;
	}

	float* GetInputGradientTensor() override
	{
		return inputTensor;
	}

	void ZeroForward() override
	{
		memset(outputTensor, 0, sizeof(float) * inputTensorHeight * outputTensorWidth);
	}

	void ZeroBackward() override
	{
		//memset(inputTensor, 0, sizeof(float) * inputTensorHeight * inputTensorWidth);
	}
	
	void Forward() override
	{
		cpuSgemmStridedBatched(
			false, false,
			outputTensorWidth, inputTensorHeight, inputTensorWidth,
			&ONEF,
			weightTensor, inputTensorWidth, inputTensorWidth * outputTensorWidth,
			inputTensor, inputTensorWidth, inputTensorHeight * inputTensorWidth,
			&ONEF,
			outputTensor, outputTensorWidth, inputTensorHeight * outputTensorWidth,
			1
		);
	}

	void Backward() override
	{
	}

	void Update() override
	{
	}
};

struct NeuralNetwork
{
	std::vector<float*> tensors;
	std::vector<Operation*> operations;

	~NeuralNetwork()
	{
		for (auto& tensor : tensors)
			delete[] tensor;
		for (auto& operation : operations)
			delete operation;
	}

	float* AddTensor(float* tensor)
	{
		tensors.emplace_back(tensor);
		return tensor;
	}

	void AddOperation(Operation* operation)
	{
		operations.emplace_back(operation);
	}

	void Forward()
	{
		for (float* tensor : tensors)
			memset(tensor, 0, sizeof(float));
		
		for (Operation* operation : operations)
			operation->Forward();
	}
};

int main()
{
	NeuralNetwork nn;
	float* inputTensor = new float[1 * 2];
	float* productTensor = nn.AddTensor(new float[1 * 3]);
	nn.AddOperation(new Linear(1, 2, 3, inputTensor, productTensor));

	inputTensor[0] = 1.0f;
	inputTensor[1] = 2.0f;
	
	nn.Forward();
	
	PrintMatrixf32(inputTensor, 1, 2, "inputTensor");
	PrintMatrixf32(productTensor, 1, 3, "productTensor");
	
	delete[] inputTensor;
	
	return 0;
}