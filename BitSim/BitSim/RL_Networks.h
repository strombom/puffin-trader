#pragma once
#include "pch.h"

#include "RL_State.h"
#include "RL_Action.h"
#include "BitBotConstants.h"


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


class RL_Networks
{
public:
    RL_Networks(void) :
        policy(TanhGaussianDistParams{ "policy", BitSim::Trader::state_dim, BitSim::Trader::action_dim }),
        vf(MultilayerPerceptron{ "vf", BitSim::Trader::state_dim, 1 }),
        vf_target(MultilayerPerceptron{ "vf_target", BitSim::Trader::state_dim, 1 }),
        qf_1(FlattenMultilayerPerceptron{ "qf_1", BitSim::Trader::state_dim + BitSim::Trader::action_dim, 1 }),
        qf_2(FlattenMultilayerPerceptron{ "vf", BitSim::Trader::state_dim + BitSim::Trader::action_dim, 1 })
    {}

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> forward_policy(torch::Tensor states);
    RL_Action get_action(RL_State state);
    RL_Action get_random_action(void);

private:
    TanhGaussianDistParams policy;
    MultilayerPerceptron vf;
    MultilayerPerceptron vf_target;
    FlattenMultilayerPerceptron qf_1;
    FlattenMultilayerPerceptron qf_2;
};
