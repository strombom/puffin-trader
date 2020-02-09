#pragma once
#include "pch.h"

#include <string>

#include "FE_Model.h"


class FE_Inference
{
public:
    FE_Inference(const std::string& path, const std::string& filename);

    torch::Tensor forward(torch::Tensor observations);

private:
    RepresentationLearner model{};
};
