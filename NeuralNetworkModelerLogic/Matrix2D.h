#pragma once
#include "Matrix.h"

class Matrix2D : public Matrix
{
public:
	Matrix2D(uint32_t rows, uint32_t columns)
		: rows(rows), columns(columns)
	{
		setSize();
	}

	void setSize() override
	{
		size = rows * columns;
		arr = new float[size];
	}

private:
	uint32_t rows;
	uint32_t columns;
};