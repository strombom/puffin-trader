#pragma once
#include "pch.h"


class FeaturePlotter
{
public:
    FeaturePlotter(const std::string& file_path, const std::string& filename);

    void plot(std::vector<double> prices, const std::string& file_path, const std::string& filename);

private:
    torch::Tensor features;

};

