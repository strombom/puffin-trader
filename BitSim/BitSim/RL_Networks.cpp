#include "pch.h"
#include "RL_Networks.h"
#include "BitBotConstants.h"


MultilayerPerceptronImpl::MultilayerPerceptronImpl(const std::string& name, int input_size, int output_size)
{
    // Hidden layers
    auto next_size = input_size;
    for (auto idx = 0; idx < BitSim::Trader::hidden_count; ++idx) {
        auto hidden_layer = register_module(name + "_linear_" + std::to_string(idx), torch::nn::Linear{ next_size, BitSim::Trader::hidden_size });
        auto activation = register_module(name + "_activation_" + std::to_string(idx), torch::nn::ReLU6{});

        layers->push_back(hidden_layer);
        layers->push_back(activation);
        
        next_size = BitSim::Trader::hidden_size;
    }

    // Output layer
    auto output_layer = register_module(name + "_linear_output", torch::nn::Linear{ BitSim::Trader::hidden_size, output_size });
    auto activation = register_module(name + "_output_activation", torch::nn::ReLU6{});

    layers->push_back(output_layer);
    layers->push_back(activation);
}

FlattenMultilayerPerceptronImpl::FlattenMultilayerPerceptronImpl(const std::string& name, int input_size, int output_size) :
    mlp(register_module(name, MultilayerPerceptron{ name + "_mlp", input_size, output_size }))
{

}

GaussianDistImpl::GaussianDistImpl(const std::string& name, int input_size, int output_size) : 
    mlp(register_module(name, MultilayerPerceptron{ name + "_mlp", input_size, output_size }))
{

}

TanhGaussianDistParamsImpl::TanhGaussianDistParamsImpl(const std::string& name, int input_size, int output_size) :
    gaussian_dist(register_module(name, GaussianDist{ name + "_gauss", input_size, output_size }))
{

}

torch::Tensor MultilayerPerceptronImpl::forward(torch::Tensor x)
{
    return layers->forward(x);
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
