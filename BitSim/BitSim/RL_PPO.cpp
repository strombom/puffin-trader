#include "pch.h"

#include "RL_PPO.h"
#include "BitBotConstants.h"


RL_PPO_ModelImpl::RL_PPO_ModelImpl(const std::string& name) :
    policy_mean(register_module(name + "_policy_mean", torch::nn::Linear{ hidden_dim, BitSim::Trader::action_dim })),
    policy_log_std(register_module(name + "_policy_log_std", torch::nn::Linear{ hidden_dim, BitSim::Trader::action_dim })),
    state_value(register_module(name + "_state_value", torch::nn::Linear{ hidden_dim, 1 }))
{
    network->push_back(register_module(name + "_actor_linear_1", torch::nn::Linear{ BitSim::Trader::state_dim, hidden_dim }));
    network->push_back(register_module(name + "_actor_dropout_1", torch::nn::Dropout{ dropout }));
    network->push_back(register_module(name + "_actor_tanh_1", torch::nn::Tanh{}));
    network->push_back(register_module(name + "_actor_linear_2", torch::nn::Linear{ hidden_dim, hidden_dim }));
    network->push_back(register_module(name + "_actor_dropout_2", torch::nn::Dropout{ dropout }));
    network->push_back(register_module(name + "_actor_tanh_2", torch::nn::Tanh{}));
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> RL_PPO_ModelImpl::forward(torch::Tensor state)
{
    const auto latent_state = network->forward(state);
    const auto value_f = state_value->forward(latent_state);
    const auto action_mean = policy_mean->forward(latent_state);
    const auto action_std = policy_log_std->forward(latent_state).exp();

    // Sample with reparametrization trick
    const auto eps = torch::normal(0.0, 1.0, action_std.sizes());
    const auto action = torch::tanh(action_mean + eps * action_std);

    // Normalize action and log_prob
    const auto log_prob = -(action - action_mean).pow(2) / (2 * action_std.pow(2)) - action_std.log() - std::log(std::sqrt(2 * M_PI));
    const auto neg_log_prob = -log_prob;

    // Entropy
    const auto entropy = 0.5 + 0.5 * std::log(2 * M_PI) + torch::log(action_std);

    return std::make_tuple(value_f, action, neg_log_prob, entropy);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> RL_PPO_ModelImpl::forward(torch::Tensor state, torch::Tensor action)
{
    const auto latent_state = network->forward(state);
    const auto value = state_value->forward(latent_state);

    // Normalize action and log_prob
    const auto log_prob = -(action - action.mean()).pow(2) / (2 * action.std().pow(2)) - action.std().log() - std::log(std::sqrt(2 * M_PI));
    const auto neg_log_prob = -log_prob;

    // Entropy
    const auto entropy = 0.5 + 0.5 * std::log(2 * M_PI) + torch::log(action.std());

    return std::make_tuple(value, action, neg_log_prob, entropy);
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
    values = torch::zeros({ BitSim::Trader::max_steps, 1 });
    neglogprobs = torch::zeros({ BitSim::Trader::max_steps, 1 });
    dones = torch::zeros({ BitSim::Trader::max_steps });
    rewards = torch::zeros({ BitSim::Trader::max_steps });
}

void RL_PPO_ReplayBuffer::clear(void)
{
    length = 0;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> RL_PPO_ReplayBuffer::sample(void)
{
    return std::make_tuple(
        states.narrow(0, 0, length),
        actions.narrow(0, 0, length),
        values.narrow(0, 0, length),
        neglogprobs.narrow(0, 0, length),
        dones.narrow(0, 0, length),
        rewards.narrow(0, 0, length)
    );
}

RL_PPO::RL_PPO(void) :
    policy(RL_PPO_Model{ "policy" })
{
    policy_optim = std::make_unique<torch::optim::Adam>(policy->parameters(), BitSim::Trader::ppo_policy_learning_rate);
}

void RL_PPO::append_to_replay_buffer(sptrRL_State current_state, sptrRL_Action action, sptrRL_State next_state, bool done)
{
    replay_buffer.dones[replay_buffer.length] = next_state->done;
    replay_buffer.rewards[replay_buffer.length] = next_state->reward;
    ++replay_buffer.length;
    last_state = next_state;
}

sptrRL_Action RL_PPO::get_action(sptrRL_State state)
{
    const auto [value_f, action, neg_log_prob, entropy] = policy->forward(state->to_tensor());

    replay_buffer.states[replay_buffer.length] = state->to_tensor();
    replay_buffer.actions[replay_buffer.length] = action;
    replay_buffer.values[replay_buffer.length] = value_f;
    replay_buffer.neglogprobs[replay_buffer.length] = neg_log_prob;

    return std::make_shared<RL_Action>(action.view({ BitSim::Trader::action_dim }));
}

sptrRL_Action RL_PPO::get_random_action(sptrRL_State state)
{
    // Not used
    return std::make_shared<RL_Action>();
}

std::array<double, 6> RL_PPO::update_model(void)
{
    auto [states, actions, values, neglogprobs, dones, rewards] = replay_buffer.sample();
    replay_buffer.clear();

    std::cout << "states " << states << std::endl;
    std::cout << "actions " << actions << std::endl;
    std::cout << "values " << values << std::endl;
    std::cout << "neglogprobs " << neglogprobs << std::endl;
    std::cout << "dones " << dones << std::endl;
    std::cout << "rewards " << rewards << std::endl;

    const auto [last_value, _action, _neg_log_prob, _entropy] = policy->forward(last_state->to_tensor());

    const auto length = rewards.size(0);
    auto next_value = last_value.item().toDouble();
    auto next_non_terminal = 1.0 - dones[length - 1].item().toDouble();
    auto advs = torch::zeros_like(rewards);
    auto lastgaelam = 0.0;

    for (auto idx = length - 1; idx >= 0; --idx) {
        if (idx < length - 1) {
            next_non_terminal = 1.0 - dones[idx + 1].item().toDouble();
            next_value = values[idx + 1].item().toDouble();
        }

        const auto delta = rewards[idx].item().toDouble() + BitSim::Trader::gamma_discount * next_value * next_non_terminal - values[idx].item().toDouble();
        lastgaelam = delta + BitSim::Trader::gamma_discount * BitSim::Trader::lam_discount * next_non_terminal * lastgaelam;
        advs[idx] = lastgaelam;
    }

    auto returns = advs + values;

    auto loss = 0.0;
    auto pg_loss = 0.0;
    auto value_loss = 0.0;
    auto entropy_mean = 0.0;
    auto approx_kl = 0.0;

    const auto n_batches = BitSim::Trader::ppo_n_updates * BitSim::Trader::ppo_n_batches;

    for (auto idx_batch = 0; idx_batch < n_batches; ++idx_batch) {
        auto indices = torch::randint(length, BitSim::Trader::ppo_batch_size, torch::TensorOptions{}.dtype(torch::ScalarType::Long));

        auto s_states = states.index(indices);
        auto s_old_actions = actions.index(indices);
        auto s_old_values = values.index(indices);
        auto s_old_neglogprobs = neglogprobs.index(indices);
        auto s_dones = dones.index(indices);
        auto s_returns = returns.index(indices);

        auto advantages = torch::empty_like(returns);
        {
            torch::NoGradGuard no_grad;
            advantages = returns - s_old_values;
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8);
        }

        policy->train();
        policy->zero_grad();

        auto [values, actions, neg_log_prob, entropy] = policy->forward(s_states, s_old_actions);
        auto [s_loss, s_pg_loss, s_value_loss, s_entropy_mean, s_approx_kl] = policy->loss(s_returns, values, neg_log_prob, entropy, advantages, s_old_values, s_old_neglogprobs);

        s_loss.backward();

        loss += s_loss.item().toDouble();
        pg_loss += s_pg_loss.item().toDouble();
        value_loss += s_value_loss.item().toDouble();
        entropy_mean += s_entropy_mean.item().toDouble();
        approx_kl += s_approx_kl.item().toDouble();
        
        torch::nn::utils::clip_grad_norm_(policy->parameters(), max_grad_norm);
        policy_optim->step();
    }

    loss /= n_batches;
    pg_loss /= n_batches;
    value_loss /= n_batches;
    entropy_mean /= n_batches;
    approx_kl /= n_batches;

    return std::array<double, 6>{ loss, pg_loss, value_loss, entropy_mean, approx_kl };
}
