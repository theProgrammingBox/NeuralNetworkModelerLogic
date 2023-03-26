#include <iostream>
#include <vector>
#include <memory>
#include <list>
#include <deque>

struct Param {
    std::shared_ptr<int> value;
    std::list<std::shared_ptr<int>>::iterator param_it;
};

class Matrix {
public:
    Matrix(Param& rows, Param& columns)
        : numRows(rows), numColumns(columns) {}

    Param& numRows;
    Param& numColumns;
};

std::vector<Matrix> initialize_matrices(int numMatrices, std::list<std::shared_ptr<int>>& parameters, std::deque<Param>& param_vec) {
    std::vector<Matrix> matrices;

    for (int i = 0; i < numMatrices; i++) {
        parameters.push_back(std::make_shared<int>(0));
        auto numRowsIt = std::prev(parameters.end());
        parameters.push_back(std::make_shared<int>(0));
        auto numColumnsIt = std::prev(parameters.end());

        param_vec.push_back({ *numRowsIt, numRowsIt });
        param_vec.push_back({ *numColumnsIt, numColumnsIt });

        matrices.push_back(Matrix(param_vec[param_vec.size() - 2], param_vec[param_vec.size() - 1]));
    }

    return matrices;
}

class OperationComponent {
public:
    OperationComponent(Matrix& left, Matrix& right, std::list<std::shared_ptr<int>>& parameters)
        : leftMatrix(left), rightMatrix(right) {
        // Merge/share the parameter pointers between the left and right matrices
        rightMatrix.numRows.value = leftMatrix.numColumns.value;
        rightMatrix.numRows.param_it = leftMatrix.numColumns.param_it;

        // Remove the extra parameter from the parameters list
        parameters.erase(rightMatrix.numColumns.param_it);
    }

    Matrix& leftMatrix;
    Matrix& rightMatrix;
};

std::vector<OperationComponent> initialize_operations(int numOperations, std::vector<Matrix>& matrices, std::list<std::shared_ptr<int>>& parameters) {
    std::vector<OperationComponent> operations;

    for (int i = 0; i < numOperations; i++) {
        int leftMatrixIndex, rightMatrixIndex;
        std::cout << "Enter the indices of the matrices for operation " << i + 1 << ": ";
        std::cin >> leftMatrixIndex >> rightMatrixIndex;

        operations.push_back(OperationComponent(matrices[leftMatrixIndex], matrices[rightMatrixIndex], parameters));
    }

    return operations;
}

int main() {
    int numMatrices, numOperations;
    std::cout << "Enter the number of matrices: ";
    std::cin >> numMatrices;
    std::cout << "Enter the number of operation components: ";
    std::cin >> numOperations;

    std::list<std::shared_ptr<int>> parameters;
    std::deque<Param> param_vec;

    auto matrices = initialize_matrices(numMatrices, parameters, param_vec);
    auto operations = initialize_operations(numOperations, matrices, parameters);

    int numUniqueDimensions = parameters.size();
    std::cout << "Number of unique dimension parameters: " << numUniqueDimensions << std::endl;

    auto paramIt = parameters.begin();
    for (int i = 0; i < numUniqueDimensions; i++) {
        int dimensionSize;
        std::cout << "Enter the size for dimension parameter " << i + 1 << ": ";
        std::cin >> dimensionSize;
        **paramIt = dimensionSize;
        ++paramIt;
    }

    std::cout << "Matrix dimensions:" << std::endl;
    for (int i = 0; i < numMatrices; i++) {
		std::cout << "Matrix " << i + 1 << ": " << *matrices[i].numRows.value << " x " << *matrices[i].numColumns.value << std::endl;
	}

	return 0;
}