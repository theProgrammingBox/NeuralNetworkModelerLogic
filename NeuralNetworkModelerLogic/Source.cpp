#include <iostream>
#include <vector>
#include <memory>

class StarterMatrix
{
public:
};

class Matrix
{
public:
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
	std::vector<std::shared_ptr<StarterMatrix>> starterMatrixes;
	std::vector<std::shared_ptr<Matrix>> matrixes;
    std::vector<std::shared_ptr<Layer>> layers;
	std::vector<std::shared_ptr<StarterMatrix>> endingMatrixes;

	void AddStarterMatrix()
	{
		starterMatrixes.push_back(std::make_shared<StarterMatrix>());
	}

	void AddEndingMatrix()
	{
		endingMatrixes.push_back(std::make_shared<StarterMatrix>());
	}

	template<typename T>
	void AddLayer()
	{
		layers.push_back(std::make_shared<T>());
	}
};

int main()
{
    NetworkModeler modeler;
	
    modeler.AddStarterMatrix();
	modeler.AddStarterMatrix();
	modeler.AddEndingMatrix();
	
    modeler.AddLayer<MatMulLayer>();
    modeler.AddLayer<MatAddLayer>();
	
	modeler.layers[0]->AssignInputMatrix(0, modeler.starterMatrixes[0]);
	modeler.layers[1]->AssignInputMatrix(0, modeler.starterMatrixes[1]);
	modeler.layers[1]->AssignInputMatrix(1, modeler.layers[0]->GetOutputMatrix(0));

	modeler.endingMatrixes[0]->AssignInputMatrix(modeler.layers[1]->GetOutputMatrix(0));

    return 0;
}
