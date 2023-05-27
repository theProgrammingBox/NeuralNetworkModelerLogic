#include <iostream>
#include <vector>

/*
TODO:
- add connections between nodes when operations are defined

- add a sort of compile, i expect this node given these other nodes (defining input and output)
- add a sort of compiled backpropagation, black list / white list, idk how its gonna work yet

- New nodes and operations
-- 
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

	return 0;
}