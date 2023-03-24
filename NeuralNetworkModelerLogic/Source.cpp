#include "iostream"
#include "vector"

class Matrix
{
public:
	uint32_t rows;
	uint32_t cols;
	uint32_t depth;
	uint32_t size;
	float* arr;
};

class OperationComponent
{
public:
	uint8_t inputCount;
	uint8_t outputCount;
	std::vector<Matrix*> inputMatrices;
	std::vector<Matrix*> outputMatrices;
	
	OperationComponent() = default;
	virtual ~OperationComponent() = default;

	virtual void SetInputMatrix(uint8_t idx, Matrix* matrix) = 0;
	virtual void SetOutputMatrix(uint8_t idx, Matrix* matrix) = 0;
};

class MatMulComponent : public OperationComponent
{
public:
	MatMulComponent()
	{
		inputCount = 2;
		outputCount = 1;
		inputMatrices.resize(inputCount);
		outputMatrices.resize(outputCount);
	}
	virtual ~MatMulComponent() = default;

	virtual void SetInputMatrix(uint8_t idx, Matrix* matrix) override
	{
		if (idx < inputCount)
		{
			inputMatrices[idx] = matrix;
		}
	}

	virtual void SetOutputMatrix(uint8_t idx, Matrix* matrix) override
	{
		if (idx < outputCount)
		{
			outputMatrices[idx] = matrix;
		}
	}
};

class Conv3DComponent : public OperationComponent
{
public:
	Conv3DComponent()
	{
		inputCount = 2;
		outputCount = 1;
		inputMatrices.resize(inputCount);
		outputMatrices.resize(outputCount);
	}
	virtual ~Conv3DComponent() = default;

	virtual void SetInputMatrix(uint8_t idx, Matrix* matrix) override
	{
		if (idx < inputCount)
		{
			inputMatrices[idx] = matrix;
		}
	}

	virtual void SetOutputMatrix(uint8_t idx, Matrix* matrix) override
	{
		if (idx < outputCount)
		{
			outputMatrices[idx] = matrix;
		}
	}
};

int main()
{
	std::vector<Matrix*> matrices;
	matrices.push_back(new Matrix());
	matrices.push_back(new Matrix());
	matrices.push_back(new Matrix());
	matrices.push_back(new Matrix());
	matrices.push_back(new Matrix());
	
	std::vector<OperationComponent*> components;
	components.push_back(new MatMulComponent());
	components.push_back(new Conv3DComponent());

	components[0]->SetInputMatrix(0, matrices[0]);
	components[0]->SetInputMatrix(1, matrices[1]);
	components[0]->SetOutputMatrix(0, matrices[2]);
	
	components[1]->SetInputMatrix(0, matrices[2]);
	components[1]->SetInputMatrix(1, matrices[3]);
	components[1]->SetOutputMatrix(0, matrices[4]);
}