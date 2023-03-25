#pragma once
#include "Component.h"

class Matrix
{
public:
	virtual ~Matrix()
	{
		delete[] arr;
	}

	virtual void setSize() = 0;

protected:
	uint32_t size;
	float* arr;
};