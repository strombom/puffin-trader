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
    RL_Networks(void);

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> forward_policy(torch::Tensor states); // action, log_prob, z, mean, std
    RL_Action get_action(RL_State state);
    RL_Action get_random_action(void);
    std::array<double, 5> update_model(int step, torch::Tensor states, torch::Tensor actions, torch::Tensor rewards, torch::Tensor next_states);

private:
    torch::Tensor log_alpha;
    torch::optim::Adam alpha_optim;
    double target_entropy;
    torch::optim::Adam qf_1_optim;
    torch::optim::Adam qf_2_optim;
    torch::optim::Adam vf_optim;
    torch::optim::Adam actor_optim;

    TanhGaussianDistParams actor;
    MultilayerPerceptron vf;
    MultilayerPerceptron vf_target;
    FlattenMultilayerPerceptron qf_1;
    FlattenMultilayerPerceptron qf_2;
};
