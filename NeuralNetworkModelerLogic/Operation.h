#pragma once
#include "TensorNode.h"

struct Operation
{
	virtual ~Operation() = default;
	virtual void Forward() = 0;
};