#include "pch.h"

#include "FE_Inference.h"


FE_Inference::FE_Inference(const std::string& path, const std::string& filename)
{
    torch::load(model, path + "\\" + filename);
    model->to(torch::DeviceType::CUDA);
    model->eval();
}

torch::Tensor FE_Inference::forward(torch::Tensor observations)
{
    observations = observations.reshape({ observations.size(0), BitSim::n_channels, 1, BitSim::feature_size }).cuda();
    auto feature = model->forward_predict(observations);
    return feature;
}
