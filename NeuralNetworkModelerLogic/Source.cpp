#include <iostream>
#include <vector>
#include <memory>

class Matrix
{
public:
};

class Layer
{
public:
    std::shared_ptr<Matrix> inputMatrix;

    void addInputMatrix(std::shared_ptr<Matrix> matrix)
    {
        inputMatrix = matrix;
    }
};

class ReluLayer : public Layer
{
public:
};

class NetworkModeler
{
public:
    std::vector<std::shared_ptr<Matrix>> matrixes;
    std::vector<std::shared_ptr<Layer>> layers;

    std::shared_ptr<Matrix> AddMatrix()
    {
        matrixes.push_back(std::make_shared<Matrix>());
        return matrixes.back();
    }

    template<typename T>
    std::shared_ptr<T> AddLayer()
    {
        layers.push_back(std::make_shared<T>());
        return std::static_pointer_cast<T>(layers.back());
    }
};

int main() {
    NetworkModeler modeler;
    auto userInputMatrix = modeler.AddMatrix();
    auto hiddenLayer = modeler.AddLayer<ReluLayer>();

    hiddenLayer->addInputMatrix(userInputMatrix);

    return 0;
}
