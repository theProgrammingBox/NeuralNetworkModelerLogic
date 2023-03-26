#include <iostream>
#include <vector>
#include <memory>

/*
TODO
0. decide on how the output matrixes of layers work.
(maybe have everything in terms of names and ids instead of a actual data)
(like this layer connect to that matrix. that matrix is "num_food" by "num_people")
0. work out the logic of compile. what should the comiler create and ask for.
0. should I used named parameters for dimentions? Then have diffrent types of matrixes with diffrent dimentions?
1. rework MatrixNode. Might just use matrix or maybe just use MatrixNode.
(the problem is compiler needs a direction to go. I can't just use the matrixes as they are.)
(or not, idk, work out compiler)
1. when adding matrixes by compiler, add/render the matrix so it renders in the front
2. work out zooming and panning. draw text might be a problem
2. work out a node editing tab/sidebar. like a node editor
2. when hovering over a matrix, highlight all of the connections it represents
*/

class Matrix
{
public:
	// unique id, like color, implement this later
	// name, like "input matrix", implement this later
	// dimensions?, work out the different types and what it means for the compiler
	//std::unique_ptr<float[]> data;	// not nessicary, if we are just using the matrix for visualization
};

class inputMatrixNode
{
public:
	std::shared_ptr<Matrix> matrix;

	// ability to create matrixes defined by user or predefined matrixes
};

class outputMatrixNode
{
public:
	std::shared_ptr<Matrix> matrix;

	// can only connect to predefined matrixes
	void AssignMatrix(std::shared_ptr<Matrix> matrix)
	{
		this->matrix = matrix;
	}
};

class Layer
{
public:
	uint32_t inputMatrixesCount;
	uint32_t outputMatrixesCount;
	std::unique_ptr<std::shared_ptr<Matrix>[]> inputMatrixes;
	std::unique_ptr<std::shared_ptr<Matrix>[]> outputMatrixes;

	void AssignInputMatrix(uint32_t index, std::shared_ptr<Matrix> matrix)
	{
		inputMatrixes[index] = matrix;
	}

	void AssignOutputMatrix(uint32_t index, std::shared_ptr<Matrix> matrix)
	{
		outputMatrixes[index] = matrix;
	}

	std::shared_ptr<Matrix> GetOutputMatrix(uint32_t index)
	{
		return outputMatrixes[index];
	}
};

class ReluLayer : public Layer
{
public:
	ReluLayer()
	{
		inputMatrixesCount = 1;
		outputMatrixesCount = 1;
		inputMatrixes = std::make_unique<std::shared_ptr<Matrix>[]>(inputMatrixesCount);
		outputMatrixes = std::make_unique<std::shared_ptr<Matrix>[]>(outputMatrixesCount);
	}
};

class MatMulLayer : public Layer
{
public:
	MatMulLayer()
	{
		inputMatrixesCount = 2;
		outputMatrixesCount = 1;
		inputMatrixes = std::make_unique<std::shared_ptr<Matrix>[]>(inputMatrixesCount);
		outputMatrixes = std::make_unique<std::shared_ptr<Matrix>[]>(outputMatrixesCount);
	}
};

class MatAddLayer : public Layer
{
public:
	MatAddLayer()
	{
		inputMatrixesCount = 2;
		outputMatrixesCount = 1;
		inputMatrixes = std::make_unique<std::shared_ptr<Matrix>[]>(inputMatrixesCount);
		outputMatrixes = std::make_unique<std::shared_ptr<Matrix>[]>(outputMatrixesCount);
	}
};

class NetworkModeler
{
public:
	std::vector<std::shared_ptr<inputMatrixNode>> inputMatrixNodes;
	std::vector<std::shared_ptr<Matrix>> matrixes;
    std::vector<std::shared_ptr<Layer>> layers;
	std::vector<std::shared_ptr<outputMatrixNode>> outputMatrixNodes;

	void AddInputMatrixNode()
	{
		inputMatrixNodes.push_back(std::make_shared<inputMatrixNode>());
	}

	void AddOutputMatrixNode()
	{
		outputMatrixNodes.push_back(std::make_shared<outputMatrixNode>());
	}

	template<typename T>
	void AddLayer()
	{
		layers.push_back(std::make_shared<T>());
	}

	void Compile()
	{
		// for matmul, if there is no weight matrix, ask the user if they want to create one
		// suggest a name for the matrix, dimensions, etc
		// same for output. all layers should generate an output matrix
		// matadd generated a bias matrix, so suggest a name with bias in it.

		// ask user to create or connect matrix to inputMatrixNode if there is no matrix in a created node
		// for outputMatrixNode, just ignore it if there is no matrix
		// maybe add a clean up function to remove all the matrixes that are not connected to anything
	}
};

int main()
{
    NetworkModeler modeler;
	
    modeler.AddInputMatrixNode();
	modeler.AddInputMatrixNode();
	
    modeler.AddLayer<MatMulLayer>();
    modeler.AddLayer<MatAddLayer>();
	
	modeler.layers[0]->AssignInputMatrix(0, modeler.inputMatrixNodes[0]->matrix);
	modeler.layers[1]->AssignInputMatrix(0, modeler.inputMatrixNodes[1]->matrix);
	modeler.layers[1]->AssignInputMatrix(1, modeler.layers[0]->GetOutputMatrix(0));
	
	modeler.AddOutputMatrixNode();

	modeler.outputMatrixNodes[0]->AssignMatrix(modeler.layers[1]->GetOutputMatrix(0));

	modeler.Compile();

    return 0;
}
