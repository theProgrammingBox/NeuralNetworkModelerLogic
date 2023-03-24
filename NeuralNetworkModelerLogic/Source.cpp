#include "Matrix2D.h"
#include "Matrix3D.h"
#include "Matrix4D.h"

#include "MatMulComponent.h"
#include "Conv3DComponent.h"

int main()
{
	std::vector<Matrix*> matrices;
	std::vector<OperationComponent*> components;
	
	matrices.push_back(new Matrix3D(16, 16, 1));
	matrices.push_back(new Matrix4D(4, 4, 1, 4));
	matrices.push_back(new Matrix3D(4, 4, 4));
	matrices.push_back(new Matrix2D(16, 4));
	matrices.push_back(new Matrix2D(16, 4));
	
	components.push_back(new MatMulComponent());
	components.push_back(new Conv3DComponent());

	components[0]->SetInputMatrix(0, matrices[0]);
	components[0]->SetInputMatrix(1, matrices[1]);
	components[0]->SetOutputMatrix(0, matrices[2]);
	
	components[1]->SetInputMatrix(0, matrices[2]);
	components[1]->SetInputMatrix(1, matrices[3]);
	components[1]->SetOutputMatrix(0, matrices[4]);
}