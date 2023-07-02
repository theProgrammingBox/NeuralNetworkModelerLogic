#include <iostream>
#include <vector>
<<<<<< < HEAD

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

	====== =

		/*
		Operations:
		- Matmul
		- Matadd
		- Concatenate
		- Transpose
		- Relu
		- Reshape
		- Softmax
		- Convolution
		*/

		/*
		Nodes:
		- ExternalInput
		- ExternalOutput
		- RecursiveLink
		- Parameter
		- Intermediate
		*/

		/*
		Compiler Notes:
		- Transpose
		-- Have a default transpose operation
		-- if matmul is followed by transpose, flip the order of the matmul
		-- if transpose is followed by matmul, set the transpose flag on the matmul
		-- if transpose is followed by transpose, remove the transpose

		- Concatenate
		-- Have a default concatenate operation
		-- if it is possible to just have the data next to each other in memory, do that

		- MatAdd
		-- Have a default matadd operation
		-- if it is possible to just add to the existing matrix without effecting other operations, do that

		- Convolution
		-- Need to know more before I can logic out optimizations tied to all this

		- Named Dimension Parameters
		-- Think this out more, should help greatly with the organization of the network
		-- Show the user the named dimension parameters, and allow them to change them

		- Reshape
		-- The only thing it does is change the shape of the tensor, data remains the same,
		it just puts a constraint on the named dimension parameters to match the new shape
		*/

		struct Tensor
	{
	};

	struct Operation
	{
	};

	struct Concatenate : Operation
	{
		Concatenate(Tensor* v1, Tensor* v2, Tensor* v3)
		{
		}
	};

	struct RecursiveLink : Operation
	{
		RecursiveLink(Tensor* v1, Tensor* v2)
		{
		}
	};

	struct MatMul : Operation
	{
		MatMul(Tensor* v1, Tensor* v2, Tensor* v3)
		{
		}
	};

	struct Relu : Operation
	{
		Relu(Tensor* v1, Tensor* v2)
		{
		}
	};

	struct MatAdd : Operation
	{
		MatAdd(Tensor* v1, Tensor* v2, Tensor* v3)
		{
		}
	};

	struct Pipeline
	{
		Pipeline(std::vector<Tensor*> inputs, std::vector<Tensor*> outputs)
		{
		}
	};

	struct NeuralNetwork
	{
		std::vector<Pipeline*> pipelines;

		NeuralNetwork(std::vector<Pipeline*> pipelines)
		{
			this->pipelines = pipelines;
		}
	};

	int main()
	{
		/*auto input = new Tensor();
		auto hidden = new Tensor();
		auto concat = new Tensor();
		auto weight1 = new Tensor();
		auto product = new Tensor();
		auto relu = new Tensor();
		auto weight2 = new Tensor();
		auto output = new Tensor();
		auto weight3 = new Tensor();
		auto presum = new Tensor();
		auto newHidden = new Tensor();

		// define external alterations so we can determin the parameter nodes
		// auto in = new ExternalInput(input);
		// auto out = new ExternalOutput(output);
		auto recursive = new RecursiveLink(newHidden, hidden);
		// concat optimizes by requiring the arrays to be next to each other in memory if possible
		// (if there is no layout conflict with other nodes)
		auto concatenate = new Concatenate(hidden, input, concat);
		auto matmul1 = new MatMul(concat, weight1, product);
		auto relu1 = new Relu(product, relu);
		auto matmul2 = new MatMul(relu, weight2, output);
		auto matmul3 = new MatMul(relu, weight3, presum);
		// matadd optimizes by not creating a new matrix containing the new sum if possible
		// (if one of the nodes is not used for further computation, just add to that node)
		auto matadd = new MatAdd(presum, hidden, newHidden);
		// auto transpose = new Transpose(temp);
		// transpose is going to be nessisary for things like attention
		// the data is going to be stored in non transposed form, but the operations are going to be altered
		// (matmul has transpose, learn about convolution, and think out others like concat)

		// first vector is all the nodes that we will feed into the pipeline
		// second vector is all the nodes that we expect to be calculated
		// all the "leaf" nodes that are not in the first vector are considered constants aka parameters
		auto forward = new Pipeline({ input }, { output });
		auto backward = new Pipeline({ output }, { input });

		auto network = new NeuralNetwork({ forward, backward });*/



		// attention weights can be optimized into a single matrix, note to compiler
		auto latent = new Tensor();
		auto queryWeight = new Tensor();
		auto keyWeight = new Tensor();
		auto valueWeight = new Tensor();
		auto query = new Tensor();
		auto key = new Tensor();
		auto keyTranspose = new Tensor();
		auto value = new Tensor();
		auto score = new Tensor();
		// define batches, define reshapes
		auto softmax = new Tensor();
		auto attention = new Tensor();
		auto attentionWeight = new Tensor();
		auto attentionOutput = new Tensor();

		auto matmul1 = new MatMul(latent, queryWeight, query);
		auto matmul2 = new MatMul(latent, keyWeight, key);
		auto matmul3 = new MatMul(latent, valueWeight, value);
		auto transpose = new Transpose(key, keyTranspose);
		auto matmul4 = new MatMul(query, keyTranspose, score);
		auto softmax1 = new Softmax(score, softmax);
		auto matmul5 = new MatMul(softmax, value, attention);
		auto matmul6 = new MatMul(attention, attentionWeight, attentionOutput);

		>>>>>> > f795b962316a94272aabbb59a149974b51052b83
			return 0;
	}