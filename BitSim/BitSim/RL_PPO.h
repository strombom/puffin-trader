#pragma once
#include "pch.h"

#include "RL_State.h"
#include "RL_Action.h"
#include "RL_Algorithm.h"


// https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/PPO_continuous.py


class RL_PPO_ReplayBuffer
{
public:
    RL_PPO_ReplayBuffer(void);

    void clear(void);
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> sample(void);

    torch::Tensor actions;
    torch::Tensor states;
    torch::Tensor log_probs;
    torch::Tensor rewards;
    torch::Tensor dones;

    int length;
};


class RL_PPO_ActorCriticImpl : public torch::nn::Module
{
public:
    RL_PPO_ActorCriticImpl(const std::string& name);

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> act(torch::Tensor state);
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> evaluate(torch::Tensor state, torch::Tensor action);
    //std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> loss(torch::Tensor reward, torch::Tensor value, torch::Tensor neg_log_prob, torch::Tensor entropy, torch::Tensor advantage, torch::Tensor old_value, torch::Tensor old_neg_log_prob);

private:
    /*
    const double dropout = 0.0;
    const double clip_range = 0.2;
    const double ent_coef = 0.0;
    const double vf_coef = 0.5;
    */
    const int hidden_dim = 64;

    torch::nn::Sequential actor;
    torch::nn::Sequential actor_mean;
    torch::nn::Sequential actor_log_std;
    torch::nn::Sequential critic;
};
TORCH_MODULE(RL_PPO_ActorCritic);


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
    RL_PPO_ActorCritic policy;
    RL_PPO_ActorCritic policy_old;
    std::unique_ptr<torch::optim::Adam> optimizer;
};
