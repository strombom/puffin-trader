#include "pch.h"

#include "RL_PPO.h"
#include "BitBotConstants.h"


RL_PPO_ModelImpl::RL_PPO_ModelImpl(const std::string& name) :
    policy_mean(register_module(name + "_policy_mean", torch::nn::Linear{ hidden_dim, BitSim::Trader::action_dim })),
    policy_log_std(register_module(name + "_policy_log_std", torch::nn::Linear{ hidden_dim, BitSim::Trader::action_dim })),
    value(register_module(name + "_value", torch::nn::Linear{ hidden_dim, 1 }))
{
    network->push_back(register_module(name + "_actor_linear_1", torch::nn::Linear{ BitSim::Trader::state_dim, hidden_dim }));
    network->push_back(register_module(name + "_actor_dropout_1", torch::nn::Dropout{ dropout }));
    network->push_back(register_module(name + "_actor_tanh_1", torch::nn::Tanh{}));
    network->push_back(register_module(name + "_actor_linear_2", torch::nn::Linear{ hidden_dim, hidden_dim }));
    network->push_back(register_module(name + "_actor_dropout_2", torch::nn::Dropout{ dropout }));
    network->push_back(register_module(name + "_actor_tanh_2", torch::nn::Tanh{}));
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> RL_PPO_ModelImpl::forward(torch::Tensor states)
{
    const auto latent_state = network->forward(states);
    const auto value_f = value->forward(latent_state);
    const auto action_mean = policy_mean->forward(latent_state);
    const auto action_log_std = policy_log_std->forward(latent_state);
    const auto action_std = action_log_std.exp();

    // Reparametrization trick
    const auto eps = torch::normal(0.0, 1.0, action_std.sizes());
    const auto action_z = action_mean + eps * action_std;

    // Normalize action and log_prob
    constexpr auto epsilon = 1e-6;
    const auto action = torch::tanh(action_z);
    const auto dist_log_prob = -(action_z - action_mean).pow(2) / (2 * action_std.pow(2)) - action_std.log() - std::log(std::sqrt(2 * M_PI));
    const auto log_prob = (dist_log_prob - (1 - action.pow(2) + epsilon).log()).sum(1, true);
    const auto neg_log_prob = -log_prob;

    // Entropy
    const auto entropy = 0.5 + 0.5 * std::log(2 * M_PI) + torch::log(action_std);

    return std::make_tuple(value_f, action, neg_log_prob, entropy);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> RL_PPO_ModelImpl::loss(torch::Tensor reward, torch::Tensor value_f, torch::Tensor neg_log_prob, torch::Tensor entropy, torch::Tensor advantages, torch::Tensor old_value_f, torch::Tensor old_neg_log_prob)
{
    const auto entropy_mean = entropy.mean();
    const auto value_f_clip = old_value_f + torch::clamp(value_f - old_value_f, -clip_range, clip_range);

    const auto value_loss1 = (value_f - reward).pow(2);
    const auto value_loss2 = (value_f_clip - reward).pow(2);

    const auto value_loss = 0.5 * torch::max(value_loss1, value_loss2).mean();
    const auto ratio = (old_neg_log_prob - neg_log_prob).exp();

    const auto pg_losses1 = -advantages * ratio;
    const auto pg_losses2 = -advantages * torch::clamp(ratio, 1.0 - clip_range, 1.0 + clip_range);
    const auto pg_loss = torch::max(pg_losses1, pg_losses2).mean();
    const auto approx_kl = 0.5 * (neg_log_prob - old_neg_log_prob).pow(2).mean();

    const auto loss = pg_loss - (entropy_mean * ent_coef) + (value_loss * vf_coef);

    return std::make_tuple(loss, pg_loss, value_loss, entropy_mean, approx_kl);
}

RL_PPO_ReplayBuffer::RL_PPO_ReplayBuffer(void) :
    length(0)
{
    states = torch::zeros({ BitSim::Trader::max_steps, BitSim::Trader::state_dim });
    actions = torch::zeros({ BitSim::Trader::max_steps, BitSim::Trader::action_dim });
    rewards = torch::zeros({ BitSim::Trader::max_steps });
    logprobs = torch::zeros({ BitSim::Trader::max_steps, BitSim::Trader::state_dim });
    dones = torch::zeros({ BitSim::Trader::max_steps });
}

void RL_PPO_ReplayBuffer::clear(void)
{
    length = 0;
}

void append_state(const RL_State& state)
{
    //states[length] = state.to_tensor();
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> RL_PPO_ReplayBuffer::sample(void)
{
    return std::make_tuple(
        states.narrow(0, 0, length), 
        actions.narrow(0, 0, length),
        rewards.narrow(0, 0, length),
        logprobs.narrow(0, 0, length),
        dones.narrow(0, 0, length)
    );
}

RL_PPO::RL_PPO(void) :
    policy(RL_PPO_Model{ "policy" })
{

}

void RL_PPO::append_to_replay_buffer(sptrRL_State current_state, sptrRL_Action action, sptrRL_State next_state, bool done)
{
    replay_buffer.states[replay_buffer.length] = current_state->to_tensor();
    replay_buffer.actions[replay_buffer.length] = action->to_tensor();
    replay_buffer.dones[replay_buffer.length] = done;
}

sptrRL_Action RL_PPO::get_action(sptrRL_State state)
{
    const auto [value_f, action, neg_log_prob, entropy] = policy->forward(state->to_tensor());
    std::cout << "action " << action.view({ BitSim::Trader::action_dim }) << std::endl;



    return std::make_shared<RL_Action>(action.view({ BitSim::Trader::action_dim }));
    /*
    const auto state_tensor = state.to_tensor().view({ 1, BitSim::Trader::state_dim });
    const auto [action, log_prob, z, mean, std] = actor->forward(state_tensor);
    return RL_Action{ action.view({ BitSim::Trader::action_dim }) };
    */
}

sptrRL_Action RL_PPO::get_random_action(sptrRL_State state)
{
    // Not used
    return std::make_shared<RL_Action>();
}

std::array<double, 6> RL_PPO::update_model(void)
{
    auto [states, actions, rewards, next_states, dones] = replay_buffer.sample();
    replay_buffer.clear();

    //auto discounted_rewards = rewards + (BitSim::Trader::gamma_discount);

    std::cout << "rewards " << rewards << std::endl;

    const auto length = rewards.size(0);
    auto discounted_reward = 0.0;
    auto discounted_rewards = torch::zeros({ rewards.size(0) });

    for (auto idx = length - 1; idx >= 0; --idx) {
        if (dones[idx].item().toInt() == 1) {
            discounted_reward = 0.0;
        }
        discounted_reward = rewards[idx].item().toDouble() + BitSim::Trader::gamma_discount * discounted_reward;
        discounted_rewards[idx] = discounted_reward;

    }

    discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-5);

    constexpr auto update_loop_count = 80;

    for (auto idx = 0; idx < update_loop_count; ++idx) {
        //auto [logprobs, state_values, dist_entropy] = policy->evaluate(states, actions);

        //auto ratios = torch::exp(logprobs - old_logprobs.detach());
    }

    /*
    # Evaluating old actions and values :
    logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            
    # Finding the ratio (pi_theta / pi_theta__old):
    ratios = torch.exp(logprobs - old_logprobs.detach())

    # Finding Surrogate Loss:
    advantages = rewards - state_values.detach()   
    surr1 = ratios * advantages
    surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
    loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            
    # take gradient step
    self.optimizer.zero_grad()
    loss.mean().backward()
    self.optimizer.step()
    */

    return std::array<double, 6>{ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };

    /*
    const auto [new_actions, log_prob, _z, _mean, _std] = actor->forward(states);
    const auto [new_next_actions, next_log_prob, _next_z, _next_mean, _next_std] = actor->forward(next_states);

    // Train Q
    const auto target_q1_ = target_soft_q1->forward(next_states, new_next_actions);
    const auto target_q2 = target_soft_q2->forward(next_states, new_next_actions);
    const auto target_q_min = torch::min(target_q1_, target_q2);
    const auto target_q_value = (rewards + BitSim::Trader::gamma_discount * target_q_min).detach();

    const auto pred_q1 = soft_q1->forward(states, actions);
    const auto pred_q2 = soft_q2->forward(states, actions);
    const auto q1_value_loss = torch::mse_loss(pred_q1, target_q_value);
    const auto q2_value_loss = torch::mse_loss(pred_q2, target_q_value);

    soft_q1_optim->zero_grad();
    q1_value_loss.backward();
    soft_q1_optim->step();

    soft_q2_optim->zero_grad();
    q2_value_loss.backward();
    soft_q2_optim->step();

    // Tune entropy
    const auto alpha_loss = -(log_alpha * (log_prob + target_entropy).detach()).mean();
    alpha_optim->zero_grad();
    alpha_loss.backward();
    alpha_optim->step();
    const auto alpha = log_alpha.exp().item().toDouble();

    // Train policy
    const auto predicted_new_q1_value = soft_q1->forward(states, new_actions);
    const auto predicted_new_q2_value = soft_q2->forward(states, new_actions);
    const auto predicted_new_q_value = torch::min(predicted_new_q1_value, predicted_new_q2_value);
    const auto actor_loss = (alpha * log_prob - predicted_new_q_value).mean();

    actor_optim->zero_grad();
    actor_loss.backward();
    actor_optim->step();

    // Soft update target
    auto param_q1 = soft_q1->parameters();
    auto param_q2 = soft_q1->parameters();
    auto target_param_q1 = target_soft_q1->parameters();
    auto target_param_q2 = target_soft_q2->parameters();
    for (auto i = 0; i < param_q1.size(); ++i) {
        target_param_q1[i].data().copy_(target_param_q1[i].data() * (1.0 - BitSim::Trader::soft_tau) + param_q1[i].data() * BitSim::Trader::soft_tau);
        target_param_q2[i].data().copy_(target_param_q2[i].data() * (1.0 - BitSim::Trader::soft_tau) + param_q2[i].data() * BitSim::Trader::soft_tau);
    }

    const auto actor_loss_d = actor_loss.item().toDouble();
    const auto alpha_loss_d = alpha_loss.item().toDouble();
    const auto q1_value_loss_d = q1_value_loss.item().toDouble();
    const auto q2_value_loss_d = q2_value_loss.item().toDouble();
    const auto total_loss = actor_loss_d + alpha_loss_d + q1_value_loss_d + q2_value_loss_d;

    const auto episode_score = rewards.sum().item().toDouble() / BitSim::Trader::batch_size;

    return std::array<double, 6>{ total_loss, actor_loss_d, alpha_loss_d, q1_value_loss_d, q2_value_loss_d, episode_score };

    */
}
