#pragma once
#include "TensorNode.h"

struct Operation
{
	virtual ~Operation() = default;
	virtual void Forward() = 0;
	virtual void Backward() = 0;
	virtual void Update(const float* learningRate) = 0;
	virtual void PrintParam() const = 0;
};