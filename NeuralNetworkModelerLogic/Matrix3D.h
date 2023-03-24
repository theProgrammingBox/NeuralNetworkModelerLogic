#pragma once
#include "Matrix.h"

class Matrix3D : public Matrix
{
public:
	Matrix3D(uint32_t rows, uint32_t columns, uint32_t channels)
		: rows(rows), columns(columns), channels(channels)
	{
		setSize();
	}

	void setSize() override
	{
		size = rows * columns * channels;
		arr = new float[size];
	}

private:
	uint32_t rows;
	uint32_t columns;
	uint32_t channels;
};