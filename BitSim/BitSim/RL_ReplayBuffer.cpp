#include "pch.h"
#include "RL_ReplayBuffer.h"


RL_ReplayBuffer::RL_ReplayBuffer(void) :
    idx(0),
    length(0)
{
    current_states = torch::zeros({ BitSim::Trader::buffer_size, BitSim::Trader::state_dim });
    actions = torch::zeros({ BitSim::Trader::buffer_size, BitSim::Trader::action_dim });
    rewards = torch::zeros({ BitSim::Trader::buffer_size, 1 });
    next_states = torch::zeros({ BitSim::Trader::buffer_size, BitSim::Trader::state_dim });
    dones = torch::zeros({ BitSim::Trader::buffer_size, 1 });
}

void RL_ReplayBuffer::append(const RL_State& current_state, const RL_Action& action, const RL_State& next_state, bool done)
{
    current_states[idx] = current_state.to_tensor();
    actions[idx] = action.to_tensor();
    rewards[idx] = current_state.reward;
    next_states[idx] = next_state.to_tensor();
    dones[idx] = done;

    idx = (idx + 1) % BitSim::Trader::buffer_size;
    length = std::min(length + 1, BitSim::Trader::buffer_size);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> RL_ReplayBuffer::sample(void)
{
    auto indices = torch::randint(length, BitSim::Trader::batch_size, torch::TensorOptions{}.dtype(torch::ScalarType::Long));
    indices = (indices + BitSim::Trader::buffer_size + idx - length).fmod(BitSim::Trader::buffer_size);

    return std::make_tuple(current_states.index(indices), actions.index(indices), rewards.index(indices), next_states.index(indices), dones.index(indices));
}
