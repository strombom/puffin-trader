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
    void append_state(sptrRL_State state);
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> sample(void);

    torch::Tensor states;
    torch::Tensor actions;
    torch::Tensor values;
    torch::Tensor neglogprobs;
    torch::Tensor dones;
    torch::Tensor rewards;

    int length;
};


class RL_PPO : public RL_Algorithm
{
public:
    RL_PPO(void);

    sptrRL_Action get_action(sptrRL_State state);
    sptrRL_Action get_random_action(sptrRL_State state);
    void append_to_replay_buffer(sptrRL_State current_state, sptrRL_Action action, sptrRL_State next_state, bool done);

    std::array<double, 6> update_model(void);


private:
    RL_PPO_ReplayBuffer replay_buffer;

    std::unique_ptr<torch::optim::Adam> policy_optim;

    RL_PPO_Model policy;

};
