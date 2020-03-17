#include "pch.h"
#include "RL_Networks.h"
#include "BitBotConstants.h"


MultilayerPerceptronImpl::MultilayerPerceptronImpl(const std::string& name, int input_size, int output_size)
{
    // Hidden layers
    auto next_size = input_size;
    for (auto idx = 0; idx < BitSim::Trader::hidden_count; ++idx) {
        auto hidden_layer = register_module(name + "_linear_" + std::to_string(idx), torch::nn::Linear{ next_size, BitSim::Trader::hidden_size });
        layers.push_back(hidden_layer);
        next_size = BitSim::Trader::hidden_size;
    }

    // Output layer
    auto output_layer = register_module(name + "_linear_output", torch::nn::Linear{ BitSim::Trader::hidden_size, output_size });
    layers.push_back(output_layer);
}

torch::Tensor MultilayerPerceptronImpl::forward(torch::Tensor x)
{
    for (auto&& layer : layers) {
        x = layer->forward(x);
    }
    return x;
}

torch::Tensor FlattenMultilayerPerceptronImpl::forward(torch::Tensor x)
{
    return mlp->forward(x);
}

torch::Tensor GaussianDistImpl::forward(torch::Tensor x)
{
    return mlp->forward(x);
}

torch::Tensor TanhGaussianDistParamsImpl::forward(torch::Tensor x)
{
    return gaussian_dist->forward(x);
}
