#pragma once
#include "Component.h"

class Matrix
{
public:
	virtual ~Matrix() = default;

	uint32_t getSize() const
	{
		return size;
	}

	float* getArr() const
	{
		return arr;
	}

	virtual void setSize() = 0;

protected:
	uint32_t size;
	float* arr;
};