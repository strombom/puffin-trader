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


class SoftQNetworkImpl : public torch::nn::Module
{
public:
    SoftQNetworkImpl(const std::string& name, int input_size, int action_size);

    torch::Tensor forward(torch::Tensor state, torch::Tensor action);

private:
    MultilayerPerceptron mlp;
};
TORCH_MODULE(SoftQNetwork);


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


class RL_SAC_ReplayBuffer
{
public:
    RL_SAC_ReplayBuffer(void);

    void append(const RL_State& current_state, const RL_Action& action, const RL_State& next_state);
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> sample(void);

private:
    torch::Tensor current_states;
    torch::Tensor actions;
    torch::Tensor rewards;
    torch::Tensor next_states;
    torch::Tensor dones;

    int idx;
    int length;
};


class RL_SAC
{
public:
    RL_SAC(void);

    RL_Action get_action(RL_State state);
    RL_Action get_random_action(void);
    std::array<double, 6> update_model(void);
    void append_to_replay_buffer(const RL_State& current_state, const RL_Action& action, const RL_State& next_state);

    void save(const std::string& filename);
    void open(const std::string& filename);

private:
    RL_SAC_ReplayBuffer replay_buffer;

    double target_entropy;
    torch::Tensor log_alpha;
    std::unique_ptr<torch::optim::Adam> actor_optim;
    std::unique_ptr<torch::optim::Adam> alpha_optim;
    std::unique_ptr<torch::optim::Adam> soft_q1_optim;
    std::unique_ptr<torch::optim::Adam> soft_q2_optim;

    TanhGaussianDistParams actor;
    SoftQNetwork soft_q1;
    SoftQNetwork soft_q2;
    SoftQNetwork target_soft_q1;
    SoftQNetwork target_soft_q2;

    int update_count;
};
