#pragma once
#include "pch.h"

#include <string>

#include "FE_Model.h"


class FE_Inference
{
public:
    FE_Inference(const std::string& file_path);



private:
    RepresentationLearner model{};
};
