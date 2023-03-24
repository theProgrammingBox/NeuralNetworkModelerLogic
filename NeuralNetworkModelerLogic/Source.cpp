#include "iostream"
#include "vector"

class Matrix
{
public:
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

	virtual void SetParameters(uint8_t idx, Matrix* matrix) = 0;
};

class MatMulComponent : public OperationComponent
{
public:
	MatMulComponent() = default;
	virtual ~MatMulComponent() = default;
};

class Conv3DComponent : public OperationComponent
{
public:
	Conv3DComponent() = default;
	virtual ~Conv3DComponent() = default;
};

int main()
{
	std::vector<Matrix*> matrices;
	std::vector<OperationComponent*> components;
	components.push_back(new MatMulComponent());
	components.push_back(new OperationComponent());
}