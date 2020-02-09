#include "pch.h"

#include "FE_Inference.h"


FE_Inference::FE_Inference(const std::string& file_path)
{
    torch::load(model, file_path);
    model->to(torch::DeviceType::CPU);
    model->eval();

}

torch::Tensor FE_Inference::forward(torch::Tensor observation)
{
    observation = observation.reshape({1, 3, 1, 128});
    std::cout << "observation: " << observation.sizes() << std::endl;
    auto feature = model->forward_predict(observation);
    std::cout << "feature: " << feature.sizes() << std::endl;
    return feature;
}
