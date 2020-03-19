#pragma once
#include "pch.h"


class MultilayerPerceptronImpl : public torch::nn::Module
{
public:
    MultilayerPerceptronImpl(const std::string& name, int input_size, int output_size);

    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Sequential layers;
};
TORCH_MODULE(MultilayerPerceptron);


class FlattenMultilayerPerceptronImpl : public torch::nn::Module
{
public:
    FlattenMultilayerPerceptronImpl(const std::string& name, int input_size, int output_size);

    torch::Tensor forward(torch::Tensor x, torch::Tensor y);

private:
    MultilayerPerceptron mlp;
};
TORCH_MODULE(FlattenMultilayerPerceptron);


class GaussianDistImpl : public torch::nn::Module
{
public:
    GaussianDistImpl(const std::string& name, int input_size, int output_size);

    std::tuple<torch::Tensor, torch::Tensor> get_dist_params(torch::Tensor x);

private:
    MultilayerPerceptron mlp;
    torch::nn::Sequential mean_layer;
    torch::nn::Sequential log_std_layer;
};
TORCH_MODULE(GaussianDist);


class TanhGaussianDistParamsImpl : public torch::nn::Module
{
public:
    TanhGaussianDistParamsImpl(const std::string& name, int input_size, int output_size);

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> forward(torch::Tensor x);

private:
    GaussianDist gaussian_dist;

};
TORCH_MODULE(TanhGaussianDistParams);
