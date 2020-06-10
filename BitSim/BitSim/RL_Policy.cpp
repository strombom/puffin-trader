#include "pch.h"

#include "RL_Policy.h"


RL_Policy::RL_Policy(const std::string& filename) :
    policy(nullptr)
{
    torch::load(policy, std::string{ BitSim::tmp_path } + "\\rl\\" + filename);
    policy->to(torch::DeviceType::CUDA);
    policy->eval();
}

std::tuple<bool, double> RL_Policy::get_action(torch::Tensor feature, double leverage)
{
    const auto state = torch::cat({ feature, torch::tensor({leverage}) }).view({ 1, BitSim::Trader::state_dim });

    const auto [next_cont_actions, next_disc_actions_idx, _next_probs, _next_log_probs] = policy->sample_action(state);

    return std::make_tuple(false, 0.0);
}
