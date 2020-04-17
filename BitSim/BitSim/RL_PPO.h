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

    torch::Tensor states;
    torch::Tensor actions;
    torch::Tensor log_probs;
    torch::Tensor rewards;
    torch::Tensor next_states;

    int length;
};


class RL_PPO_ActorImpl : public torch::nn::Module
{
public:
    RL_PPO_ActorImpl(const std::string& name);

    std::tuple< torch::Tensor, torch::Tensor> action(torch::Tensor state);
    torch::Tensor log_prob(torch::Tensor state, torch::Tensor action);

private:
    const int hidden_dim = 64;

    torch::nn::Sequential actor;
    torch::nn::Sequential actor_mean;
    torch::nn::Sequential actor_log_std;
};
TORCH_MODULE(RL_PPO_Actor);


class RL_PPO_CriticImpl : public torch::nn::Module
{
public:
    RL_PPO_CriticImpl(const std::string& name);

    torch::Tensor forward(torch::Tensor x);

private:
    const int hidden_dim = 64;

    torch::nn::Sequential critic;
};
TORCH_MODULE(RL_PPO_Critic);


class RL_PPO : public RL_Algorithm
{
public:
    RL_PPO(void);

    sptrRL_Action get_action(sptrRL_State state);
    sptrRL_Action get_random_action(sptrRL_State state);
    void append_to_replay_buffer(sptrRL_State current_state, sptrRL_Action action, sptrRL_State next_state);

    std::array<double, 6> update_model(void);

private:
    RL_PPO_ReplayBuffer replay_buffer;
    RL_PPO_Actor actor;
    RL_PPO_Critic critic;
    std::unique_ptr<torch::optim::Adam> optimizer_actor;
    std::unique_ptr<torch::optim::Adam> optimizer_critic;
};
