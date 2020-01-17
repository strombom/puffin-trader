
#include "Utils.h"

#pragma warning(push, 0)
#pragma warning(disable: 4146)
#include <torch/script.h>
#pragma warning(pop)


constexpr auto default_path = "C:\\development\\github\\puffin-trader\\tmp\\";


void Utils::save_tensor(const torch::Tensor& tensor, const std::string& filepath)
{   
    auto bytes = torch::jit::pickle_save(tensor);
    auto fout = std::ofstream{ default_path + filepath, std::ios::out | std::ios::binary };
    fout.write(bytes.data(), bytes.size());
    fout.close();
}
