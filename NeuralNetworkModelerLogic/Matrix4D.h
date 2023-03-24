#pragma once
#include "Matrix.h"

class Matrix4D : public Matrix
{
public:
	Matrix4D(uint32_t rows, uint32_t columns, uint32_t channels, uint32_t batch)
		: rows(rows), columns(columns), channels(channels), batch(batch)
	{
		setSize();
	}

	void setSize() override
	{
		size = rows * columns * channels * batch;
		arr = new float[size];
	}

private:
	uint32_t rows;
	uint32_t columns;
	uint32_t channels;
	uint32_t batch;
};