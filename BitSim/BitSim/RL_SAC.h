#pragma once
#include "pch.h"

#include "RL_State.h"
#include "RL_Action.h"
#include "RL_Algorithm.h"
#include "BitLib/BitBotConstants.h"


class QNetworkImpl : public torch::nn::Module
{
public:
    QNetworkImpl(const std::string& name);

    torch::Tensor forward(torch::Tensor state); // , torch::Tensor action);

private:
    torch::nn::Sequential layers;
};
TORCH_MODULE(QNetwork);


class PolicyNetworkImpl : public torch::nn::Module
{
public:
    PolicyNetworkImpl(const std::string& name);

    torch::Tensor forward(torch::Tensor state);
    //std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(torch::Tensor state);
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> sample_action(torch::Tensor state);

private:
    torch::nn::Sequential policy;
    //torch::nn::Sequential policy_mean;
    //torch::nn::Sequential policy_log_std;
    torch::nn::Sequential policy_discrete;
};
TORCH_MODULE(PolicyNetwork);


class RL_SAC_ReplayBuffer
{
public:
    RL_SAC_ReplayBuffer(void);

    void append(sptrRL_State current_state, sptrRL_Action action, sptrRL_State next_state);
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> sample(void);

private:
    torch::Tensor current_states;
    //torch::Tensor cont_actions;
    torch::Tensor disc_actions_idx;
    torch::Tensor rewards;
    torch::Tensor next_states;
    torch::Tensor dones;

    int idx;
    int length;
};


class RL_SAC : public RL_Algorithm
{
public:
    RL_SAC(void);

    sptrRL_Action get_action(sptrRL_State state);
    sptrRL_Action get_random_action(sptrRL_State state);
    std::array<double, 6> update_model(void);
    void append_to_replay_buffer(sptrRL_State current_state, sptrRL_Action action, sptrRL_State next_state);

    void save(const std::string& path, const std::string& name);
    void open(const std::string& path, const std::string& name);

private:
    RL_SAC_ReplayBuffer replay_buffer;

    double target_entropy;
    double alpha;
    torch::Tensor log_alpha;
    std::unique_ptr<torch::optim::Adam> policy_optim;
    std::unique_ptr<torch::optim::Adam> q1_optim;
    std::unique_ptr<torch::optim::Adam> q2_optim;
    std::unique_ptr<torch::optim::Adam> alpha_optim;

    PolicyNetwork policy;
    QNetwork q1;
    QNetwork q2;
    QNetwork target_q1;
    QNetwork target_q2;
};
