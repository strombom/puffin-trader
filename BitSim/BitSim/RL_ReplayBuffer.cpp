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
