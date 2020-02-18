#include "pch.h"

#include "Utils.h"

#include "torch/serialize.h"


void Utils::save_tensor(const torch::Tensor& tensor, const std::string& path, const std::string& filename)
{
    torch::save(tensor, path + "\\" + filename);
}

torch::Tensor Utils::load_tensor(const std::string& path, const std::string& filename)
{
    auto tensor = std::vector<torch::Tensor>{};
    torch::load(tensor, path + "\\" + filename);
    return tensor[0];
}
