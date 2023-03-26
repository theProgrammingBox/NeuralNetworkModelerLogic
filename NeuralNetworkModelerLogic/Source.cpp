#include <iostream>
#include <vector>
#include <memory>

class Matrix {
public:
    Matrix(std::shared_ptr<int> rows, std::shared_ptr<int> columns)
        : numRows(rows), numColumns(columns) {}

    std::shared_ptr<int> numRows;
    std::shared_ptr<int> numColumns;
};

class OperationComponent {
public:
    OperationComponent(std::shared_ptr<Matrix> left, std::shared_ptr<Matrix> right)
        : leftMatrix(left), rightMatrix(right) {}

    std::shared_ptr<Matrix> leftMatrix;
    std::shared_ptr<Matrix> rightMatrix;
};

int main() {
    int numMatrices, numOperations;
    std::cout << "Enter the number of matrices: ";
    std::cin >> numMatrices;
    std::cout << "Enter the number of operation components: ";
    std::cin >> numOperations;

    std::vector<std::shared_ptr<int>> parameters;
    std::vector<std::shared_ptr<Matrix>> matrices;

    for (int i = 0; i < numMatrices; i++) {
        parameters.push_back(std::make_shared<int>(0));
        parameters.push_back(std::make_shared<int>(0));
        matrices.push_back(std::make_shared<Matrix>(parameters[2 * i], parameters[2 * i + 1]));
    }

    std::vector<OperationComponent> operations;
    for (int i = 0; i < numOperations; i++) {
        int leftMatrixIndex, rightMatrixIndex;
        std::cout << "Enter the indices of the matrices for operation " << i + 1 << ": ";
        std::cin >> leftMatrixIndex >> rightMatrixIndex;

        // Merge/share the parameter pointers between the left and right matrices
        matrices[rightMatrixIndex]->numRows = matrices[leftMatrixIndex]->numColumns;
        operations.push_back(OperationComponent(matrices[leftMatrixIndex], matrices[rightMatrixIndex]));
    }

    // Remove duplicate shared_ptrs from the parameters vector
    std::vector<std::shared_ptr<int>> uniqueParameters;
    for (const auto& param : parameters) {
        if (std::find(uniqueParameters.begin(), uniqueParameters.end(), param) == uniqueParameters.end()) {
            uniqueParameters.push_back(param);
        }
    }

    int numUniqueDimensions = uniqueParameters.size();
    std::cout << "Number of unique dimension parameters: " << numUniqueDimensions << std::endl;

    for (int i = 0; i < numUniqueDimensions; i++) {
        int dimensionSize;
        std::cout << "Enter the size for dimension parameter " << i + 1 << ": ";
        std::cin >> dimensionSize;
        *uniqueParameters[i] = dimensionSize;
    }

    return 0;
}
