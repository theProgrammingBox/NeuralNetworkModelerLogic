#include <iostream>
#include <memory>

class DimentionData {
public:
	virtual uint32_t GetTotalSize() const = 0;
};

class Dimention2D : public DimentionData {
public:
    Dimention2D() {
        numRows = std::make_shared<uint32_t>(0);
		numColumns = std::make_shared<uint32_t>(0);
    }

	uint32_t GetTotalSize() const override {
		return *numRows * *numColumns;
	}
    
	std::shared_ptr<uint32_t>& GetNumRows() {
		return numRows;
	}

	std::shared_ptr<uint32_t>& GetNumColumns() {
		return numColumns;
	}
    
private:
    std::shared_ptr<uint32_t> numRows;
	std::shared_ptr<uint32_t> numColumns;
};

class Matrix {
public:
	Matrix(DimentionData dimentionData) {
		totalSize = dimentionData.GetTotalSize();
		data = std::make_unique<float[]>(totalSize);
	}

    float Get(uint32_t index) const {
        return data[index];
    }

    void Set(uint32_t index, float value) {
        data[index] = value;
    }

	uint32_t GetTotalSize() const {
		return totalSize;
	}

protected:
    uint32_t totalSize;
    std::unique_ptr<float[]> data;
};

class Matrix2D : public DataType {
public:
    Matrix2D(uint32_t rows, uint32_t columns)
        : numRows(rows), numColumns(columns) {
        totalSize = rows * columns;
        data = std::make_unique<float[]>(totalSize);
    }

private:
    uint32_t numRows;
    uint32_t numColumns;
};

class OperatorComponent {
public:
    virtual ~OperatorComponent() {}
    virtual void ConnectInput(uint32_t index, std::shared_ptr<DataType> input) = 0;
    virtual void ConnectOutput(uint32_t index, std::shared_ptr<DataType> output) = 0;
    virtual void Compute() = 0;
    virtual void InitializeOutput() = 0;

protected:
    uint32_t numInputs;
    uint32_t numOutputs;
    std::unique_ptr<std::shared_ptr<DataType>[]> inputArray;
    std::unique_ptr<std::shared_ptr<DataType>[]> outputArray;
};

class ReluComponent : public OperatorComponent {
public:
    ReluComponent() {
        numInputs = 1;
        numOutputs = 1;
        inputArray = std::make_unique<std::shared_ptr<DataType>[]>(numInputs);
        outputArray = std::make_unique<std::shared_ptr<DataType>[]>(numOutputs);
    }

    void ConnectInput(uint32_t index, std::shared_ptr<DataType> input) override {
        inputArray[index] = input;
        InitializeOutput();
    }

    void ConnectOutput(uint32_t index, std::shared_ptr<DataType> output) override {
        outputArray[index] = output;
    }

    void InitializeOutput() override {
        if (inputArray[0]) {
            outputArray[0] = std::make_shared<Matrix2D>(inputArray[0]->GetTotalSize(), 1);
        }
    }

    void Compute() override {
        for (uint32_t i = 0; i < inputArray[0]->GetTotalSize(); i++) {
            float inputValue = inputArray[0]->Get(i);
            float outputValue = inputValue > 0 ? inputValue : 0;
            outputArray[0]->Set(i, outputValue);
        }
    }
};

int main() {
    auto inputMatrix = std::make_shared<Matrix2D>(3, 3);
    auto reluComponent = std::make_shared<ReluComponent>();

    reluComponent->ConnectInput(0, inputMatrix);

    return 0;
}
