#pragma once
#include "pch.h"

#include "RL_State.h"
#include "RL_Action.h"
#include "RL_Algorithm.h"


// https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/PPO_continuous.py


class RL_PPO_ModelImpl : public torch::nn::Module
{
public:
    RL_PPO_ModelImpl(const std::string& name);

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> loss(torch::Tensor reward, torch::Tensor value_f, torch::Tensor neg_log_prob, torch::Tensor entropy, torch::Tensor advantages, torch::Tensor old_value_f, torch::Tensor old_neg_log_prob);
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> forward(torch::Tensor states);

private:
    const double dropout = 0.0;
    const double clip_range = 0.2;
    const double ent_coef = 0.0;
    const double vf_coef = 0.5;
    const int hidden_dim = 64;

    torch::nn::Sequential network;
    torch::nn::Linear policy_mean;
    torch::nn::Linear policy_log_std;
    torch::nn::Linear value;
};
TORCH_MODULE(RL_PPO_Model);


class RL_PPO_ReplayBuffer
{
public:
    RL_PPO_ReplayBuffer(void);

    void clear(void);
    void append_state(const RL_State& state);
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> sample(void);

    torch::Tensor states;
    torch::Tensor actions;
    torch::Tensor logprobs;
    torch::Tensor rewards;
    torch::Tensor dones;

    int length;
};


class RL_PPO : public RL_Algorithm
{
public:
    RL_PPO(void);

    RL_Action get_action(const RL_State& state);
    RL_Action get_random_action(const RL_State& state);
    void append_to_replay_buffer(const RL_State& current_state, const RL_Action& action, const RL_State& next_state, bool done);

    std::array<double, 6> update_model(void);


private:
    RL_PPO_ReplayBuffer replay_buffer;

    std::unique_ptr<torch::optim::Adam> policy_optim;

    RL_PPO_Model policy;

};
