#pragma once
#include "pch.h"

#include <string>


class Utils
{
public:

    static void save_tensor(const torch::Tensor &tensor, const std::string& filepath);

private:

};
