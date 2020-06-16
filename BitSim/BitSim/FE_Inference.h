#pragma once
#include "pch.h"

#include "FE_Model.h"

#include <string>


class FE_Inference
{
public:
    FE_Inference(const std::string& path, const std::string& filename);

    torch::Tensor forward(torch::Tensor observations);

private:
    RepresentationLearner model;
};

using sptrFE_Inference = std::shared_ptr<FE_Inference>;
