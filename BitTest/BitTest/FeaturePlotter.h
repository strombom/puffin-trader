#pragma once
#include "pch.h"


class FeaturePlotter
{
public:
    FeaturePlotter(const std::string& file_path, const std::string& filename);

    void plot(const std::string& file_path, const std::string& filename);

private:
    torch::Tensor features;

};

