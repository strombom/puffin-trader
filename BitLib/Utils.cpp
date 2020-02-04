#include "pch.h"

#include "Utils.h"


constexpr auto default_path = "C:\\development\\github\\puffin-trader\\tmp\\";


void Utils::save_tensor(const torch::Tensor& tensor, const std::string& filepath)
{   
    auto bytes = torch::jit::pickle_save(tensor);
    auto fout = std::ofstream{ default_path + filepath, std::ios::out | std::ios::binary };
    fout.write(bytes.data(), bytes.size());
    fout.close();
}
