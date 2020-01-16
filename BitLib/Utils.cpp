
#include "Utils.h"

#include <torch/script.h>


void Utils::save_tensor(const torch::Tensor& tensor, const std::string& filepath)
{
    auto bytes = torch::jit::pickle_save(tensor);
    auto fout = std::ofstream{ filepath, std::ios::out | std::ios::binary };
    fout.write(bytes.data(), bytes.size());
    fout.close();
}
