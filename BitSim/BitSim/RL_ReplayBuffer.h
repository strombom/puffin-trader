#pragma once
#include "pch.h"

#include "BitBotConstants.h"
#include "RL_State.h"
#include "RL_Action.h"


class RL_ReplayBuffer
{
public:
    RL_ReplayBuffer(void) :
        idx(0),
        length(0) {}

    void append(RL_State current_state, RL_State next_state, RL_Action action);
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> sample(void);

private:
    RL_State current_states[BitSim::Trader::buffer_size];
    RL_State next_states[BitSim::Trader::buffer_size];
    RL_Action actions[BitSim::Trader::buffer_size];

    int idx;
    int length;
};
