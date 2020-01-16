#pragma once

#include <string>

#pragma warning(push, 0)
#pragma warning(disable: 4146)
#include <torch/torch.h>
#pragma warning(pop)


class Utils
{
public:

    static void save_tensor(const torch::Tensor &tensor, const std::string& filepath);

private:

};
