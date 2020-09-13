#include "pch.h"

#include "RL_Policy.h"


RL_Policy::RL_Policy(const std::string& filename) : 
    policy(PolicyNetwork{ "policy" })
{
    torch::load(policy, std::string{ BitSim::tmp_path } + "\\rl\\" + filename);
    policy->to(torch::DeviceType::CUDA);
    policy->eval();
}

//std::tuple<bool, double> RL_Policy::get_action(torch::Tensor feature, double leverage)
std::tuple<bool, double> RL_Policy::get_action(torch::Tensor feature, double leverage)
{
    const auto state = torch::cat({ feature, torch::tensor({leverage}).cuda() }).view({ 1, BitSim::Trader::state_dim });
    //const auto [next_cont_actions, next_disc_actions_idx, _next_probs, _next_log_probs] = policy->sample_action(state);
    const auto [next_cont_actions, next_disc_actions_idx, _next_probs, _next_log_probs] = policy->sample_action(state);
    const auto desired_stop_loss = next_cont_actions[0].item().toDouble();
    //const auto place_order = next_disc_actions_idx[0].item().toInt() > 0;
    const auto buy = next_disc_actions_idx[0].item().toInt() == 0;
    //return std::make_tuple(place_order, desired_leverage);
    return std::make_tuple(buy, desired_stop_loss);
}
