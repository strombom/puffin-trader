#include "pch.h"

#include "FE_Inference.h"


FE_Inference::FE_Inference(const std::string& file_path)
{
    torch::load(model, file_path);
    model->to(torch::DeviceType::CPU);
    model->eval();

}

torch::Tensor FE_Inference::forward(torch::Tensor observations)
{
    observations = observations.reshape({ observations.size(0), BitSim::n_channels, 1, BitSim::feature_size });
    std::cout << "observation: " << observations.sizes() << std::endl;
    auto feature = model->forward_predict(observations);
    std::cout << "feature: " << feature.sizes() << std::endl;
    return feature;
}
