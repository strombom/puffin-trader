#include "pch.h"

#include "BitLib/Utils.h"

#include "torch/serialize.h"


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
