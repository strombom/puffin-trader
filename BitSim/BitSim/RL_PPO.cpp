#include "pch.h"

#include "RL_PPO.h"
#include "BitBotConstants.h"


RL_PPO_ReplayBuffer::RL_PPO_ReplayBuffer(void) :
    length(0)
{
    actions = torch::zeros({ BitSim::Trader::max_steps, BitSim::Trader::action_dim });
    states = torch::zeros({ BitSim::Trader::max_steps, BitSim::Trader::state_dim });
    log_probs = torch::zeros({ BitSim::Trader::max_steps, 1 });
    rewards = torch::zeros({ BitSim::Trader::max_steps, 1 });
    dones = torch::zeros({ BitSim::Trader::max_steps });
}

void RL_PPO_ReplayBuffer::clear(void)
{
    length = 0;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> RL_PPO_ReplayBuffer::sample(void)
{
    return std::make_tuple(
        states.narrow(0, 0, length),
        actions.narrow(0, 0, length),
        log_probs.narrow(0, 0, length),
        rewards.narrow(0, 0, length),
        dones.narrow(0, 0, length)
    );
}

RL_PPO_ActorCriticImpl::RL_PPO_ActorCriticImpl(const std::string& name)
{
    actor->push_back(register_module(name + "_actor_linear_1", torch::nn::Linear{ BitSim::Trader::state_dim, hidden_dim }));
    actor->push_back(register_module(name + "_actor_tanh_1", torch::nn::Tanh{}));
    actor->push_back(register_module(name + "_actor_linear_2", torch::nn::Linear{ hidden_dim, hidden_dim / 2 }));
    actor->push_back(register_module(name + "_actor_tanh_2", torch::nn::Tanh{}));

    actor_mean->push_back(register_module(name + "_actor_mean_linear_1", torch::nn::Linear{ hidden_dim / 2, BitSim::Trader::action_dim }));
    actor_mean->push_back(register_module(name + "_actor_mean_tanh_1", torch::nn::Tanh{}));

    actor_log_std->push_back(register_module(name + "_actor_log_std_linear_1", torch::nn::Linear{ hidden_dim / 2, BitSim::Trader::action_dim }));
    actor_log_std->push_back(register_module(name + "_actor_log_std_tanh_1", torch::nn::Tanh{}));

    critic->push_back(register_module(name + "_critic_linear_1", torch::nn::Linear{ BitSim::Trader::state_dim, hidden_dim }));
    critic->push_back(register_module(name + "_critic_tanh_1", torch::nn::Tanh{}));
    critic->push_back(register_module(name + "_critic_linear_2", torch::nn::Linear{ hidden_dim, hidden_dim / 2 }));
    critic->push_back(register_module(name + "_critic_tanh_2", torch::nn::Tanh{}));
    critic->push_back(register_module(name + "_critic_linear_3", torch::nn::Linear{ hidden_dim / 2, 1 }));
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> RL_PPO_ActorCriticImpl::act(torch::Tensor state)
{
    const auto actor_latent = actor->forward(state);
    const auto state_value = critic->forward(state);
    const auto action_mean = actor_mean->forward(actor_latent);
    const auto action_log_std = torch::ones({ 1, 1 }) * BitSim::Trader::ppo_action_std; // actor_log_std->forward(actor_latent);
    const auto action_std = action_log_std.exp();

    // Sample with reparametrization trick
    const auto eps = torch::normal(0.0, 1.0, action_std.sizes());
    const auto action = action_mean + eps * action_std;

    // Log prob and entropy
    const auto log_prob = -(action - action_mean).pow(2) / (2 * action_std.pow(2)) - action_log_std - std::log(std::sqrt(2 * M_PI));

    return std::make_tuple(state_value, action.detach(), log_prob);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> RL_PPO_ActorCriticImpl::evaluate(torch::Tensor state, torch::Tensor action)
{
    const auto actor_latent = actor->forward(state);
    const auto state_value = critic->forward(state);
    const auto action_mean = actor_mean->forward(actor_latent);
    const auto action_log_std = torch::ones({ 1, 1 }) * BitSim::Trader::ppo_action_std; // actor_log_std->forward(actor_latent);
    const auto action_std = action_log_std.exp();

    // Log prob and entropy
    const auto log_prob = -(action - action_mean).pow(2) / (2 * action_std.pow(2)) - action_log_std - std::log(std::sqrt(2 * M_PI));
    const auto entropy = 0.5 + 0.5 * std::log(2 * M_PI) + torch::log(action_std);

    return std::make_tuple(state_value, log_prob, entropy);
}

RL_PPO::RL_PPO(void) :
    policy(RL_PPO_ActorCritic{ "policy" }),
    policy_old(RL_PPO_ActorCritic{ "policy_old" })
{
    optimizer = std::make_unique<torch::optim::Adam>(policy->parameters(), BitSim::Trader::ppo_policy_learning_rate);
}

void RL_PPO::append_to_replay_buffer(sptrRL_State current_state, sptrRL_Action action, sptrRL_State next_state, bool done)
{
    replay_buffer.rewards[replay_buffer.length] = next_state->reward;
    replay_buffer.dones[replay_buffer.length] = next_state->done;
    ++replay_buffer.length;
}

sptrRL_Action RL_PPO::get_action(sptrRL_State state)
{
    const auto [_state_value, action, log_prob] = policy_old->act(state->to_tensor());

    replay_buffer.states[replay_buffer.length] = state->to_tensor()[0];
    replay_buffer.actions[replay_buffer.length] = action[0];
    replay_buffer.log_probs[replay_buffer.length] = log_prob[0];

    return std::make_shared<RL_Action>(action.view({ BitSim::Trader::action_dim }));
}

sptrRL_Action RL_PPO::get_random_action(sptrRL_State state)
{
    // Not used
    return std::make_shared<RL_Action>();
}

std::array<double, 6> RL_PPO::update_model(void)
{
    auto [states, actions, log_probs, rewards, dones] = replay_buffer.sample();
    replay_buffer.clear();

    // Rewards
    const auto length = rewards.size(0);
    
    auto norm_rewards = torch::empty_like(rewards);
    auto discounted_reward = 0.0;
    for (auto idx = length - 1; idx >= 0; --idx) {
        if (dones[idx].item().toBool()) {
            discounted_reward = 0.0;
        }
        discounted_reward = rewards[idx].item().toDouble() + BitSim::Trader::gamma_discount * discounted_reward;
        norm_rewards[idx] = discounted_reward;
    }
    norm_rewards = (norm_rewards - norm_rewards.mean()) / (norm_rewards.std() + 1e-5);

    auto total_loss = 0.0;
    auto pg_loss = 0.0;
    auto value_loss = 0.0;
    auto entropy_mean = 0.0;
    auto approx_kl = 0.0;

    for (auto idx_epoch = 0; idx_epoch < BitSim::Trader::ppo_update_epochs; ++idx_epoch) {
        auto b_old_states = states.narrow(0, 0, length).detach();
        auto b_old_actions = actions.narrow(0, 0, length).detach();
        auto b_old_log_probs = log_probs.narrow(0, 0, length).detach();

        policy->train();
        {
            auto auto_grad_guard = torch::AutoGradMode{ true };
            auto [state_values, log_probs, entropies] = policy->evaluate(b_old_states, b_old_actions);

            auto ratios = torch::exp(log_probs - b_old_log_probs.detach());
            auto advantages = norm_rewards - state_values.detach();
            auto surr1 = ratios * advantages;
            auto surr2 = torch::clamp(ratios, 1.0 - BitSim::Trader::ppo_eps_clip, 1.0 + BitSim::Trader::ppo_eps_clip) * advantages;
            auto loss = -torch::min(surr1, surr2) + 0.5 * torch::mse_loss(state_values, norm_rewards) - 0.01 * entropies;

            optimizer->zero_grad();
            loss.mean().backward();
            optimizer->step();

            total_loss += loss.mean().item().toDouble();
            entropy_mean += entropies.mean().item().toDouble();
        }
    }

    auto policy_params = policy->parameters();
    auto policy_old_params = policy_old->parameters();
    for (auto i = 0; i < policy_params.size(); ++i) {
        policy_old_params[i].data().copy_(policy_params[i].data());
    }

    return std::array<double, 6>{ total_loss, pg_loss, value_loss, entropy_mean, approx_kl };
}
