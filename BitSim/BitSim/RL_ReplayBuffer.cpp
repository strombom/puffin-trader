#include "pch.h"
#include "RL_ReplayBuffer.h"


void RL_ReplayBuffer::append(RL_State current_state, RL_State next_state, RL_Action action)
{
    current_states[idx] = current_state;
    next_states[idx] = next_state;
    actions[idx] = action;

    idx = (idx + 1) % BitSim::Trader::buffer_size;
    length = std::min(length + 1, BitSim::Trader::buffer_size);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> RL_ReplayBuffer::sample(void)
{
    // states, actions, rewards, next_states
    auto a = torch::zeros(1);
    auto b = torch::zeros(1);
    auto c = torch::zeros(1);
    auto d = torch::zeros(1);
    return std::make_tuple(a, b, c, d);
}
