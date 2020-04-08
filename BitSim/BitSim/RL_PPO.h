#pragma once
#include "pch.h"

#include "RL_State.h"
#include "RL_Action.h"

// https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/PPO_continuous.py


class RL_PPO_ActorCriticImpl : public torch::nn::Module
{
public:
    RL_PPO_ActorCriticImpl(const std::string& name);

    torch::Tensor act(torch::Tensor x);
    torch::Tensor evaluate(torch::Tensor x);

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

    void append(const RL_State& current_state, const RL_Action& action, const RL_State& next_state, bool done);
    void clear(void);
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> sample(void);

private:
    torch::Tensor current_states;
    torch::Tensor actions;
    torch::Tensor rewards;
    torch::Tensor next_states;
    torch::Tensor dones;

    int length;
};


class RL_PPO
{
public:
    RL_PPO(void);

    RL_Action get_action(RL_State state);
    RL_Action get_random_action(void);
    void append_to_replay_buffer(const RL_State& current_state, const RL_Action& action, const RL_State& next_state, bool done);

    std::array<double, 6> update_model(void);


private:
    RL_PPO_ReplayBuffer replay_buffer;

    std::unique_ptr<torch::optim::Adam> policy_optim;

    RL_PPO_ActorCritic policy;
    RL_PPO_ActorCritic policy_old;

};
