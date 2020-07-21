#include "pch.h"

#include "Utils.h"

#include "torch/serialize.h"
#include <iterator>


void Utils::save_tensor(const torch::Tensor& tensor, const std::string& path, const std::string& filename)
{
    torch::save(tensor.cpu(), path + "\\" + filename);
}

torch::Tensor Utils::load_tensor(const std::string &path, const std::string &filename)
{
    auto tensor = std::vector<torch::Tensor>{};
    torch::load(tensor, path + "\\" + filename);
    return tensor[0];
}

double Utils::random(double min, double max)
{
    static auto random_generator = std::mt19937{ std::random_device{}() };
    return std::uniform_real_distribution<double>{min, max}(random_generator);
}

int Utils::random(int min, int max)
{
    static auto random_generator = std::mt19937{ std::random_device{}() };
    return std::uniform_int_distribution<int>{min, max}(random_generator);
}

double Utils::random_choice(std::vector<double> choices)
{
    static auto random_generator = std::mt19937{ std::random_device{}() };
    const auto idx = std::uniform_int_distribution<int>{ 0, (int)choices.size() - 1 }(random_generator);
    return choices[idx];
}
