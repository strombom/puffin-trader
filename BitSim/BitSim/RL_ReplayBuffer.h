#pragma once
#include "pch.h"

#include "BitBotConstants.h"
#include "RL_State.h"
#include "RL_Action.h"


class RL_ReplayBuffer
{
public:
    RL_ReplayBuffer(void);

    void append(const RL_State& current_state, const RL_Action& action, const RL_State& next_state);
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> sample(void);

private:
    torch::Tensor current_states;
    torch::Tensor actions;
    torch::Tensor rewards;
    torch::Tensor next_states;

    int idx;
    int length;
};
