#include <iostream>
#include <vector>
#include <memory>

/*
TODO
0. rework MatrixNode. Might just use matrix or maybe just use MatrixNode.
1. work out the logic of compile. what should the comiler create and ask for.
*/

class Matrix
{
public:
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
		// compile layers
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
