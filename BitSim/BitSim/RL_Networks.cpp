#include "pch.h"
#include "RL_Networks.h"

MultilayerPerceptronImpl::MultilayerPerceptronImpl(const std::string& name, int input_size, int output_size, int layer_count)
{
    auto in_size = input_size;
    auto out_size = output_size;
    for (auto idx = 0; idx < layer_count; ++idx) {

        auto layer = register_module(name + "_linear_" + std::to_string(idx), torch::nn::Linear{ in_size, out_size });
        linear_layers.push_back(layer);

        in_size = output_size;
    }
}
