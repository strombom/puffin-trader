#pragma once
#include "pch.h"

#include "RL_State.h"
#include "RL_Action.h"
#include "RL_Algorithm.h"


// https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/PPO_continuous.py


class RL_PPO_ActorCriticImpl : public torch::nn::Module
{
public:
    RL_PPO_ActorCriticImpl(const std::string& name);

    torch::Tensor act(torch::Tensor state);
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> evaluate(torch::Tensor states, torch::Tensor actions);

private:
    torch::nn::Sequential actor;
    torch::nn::Sequential critic;
    torch::Tensor action_var;
};
TORCH_MODULE(RL_PPO_ActorCritic);


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

    RL_PPO_ActorCritic policy;
    RL_PPO_ActorCritic policy_old;

};
