#include "pch.h"

#include "Utils.h"


void Utils::save_tensor(const torch::Tensor& tensor, const std::string& path, const std::string& filename)
{   
    auto bytes = torch::jit::pickle_save(tensor);
    auto fout = std::ofstream{ path + "\\" + filename, std::ios::out | std::ios::binary };
    fout.write(bytes.data(), bytes.size());
    fout.close();
}
