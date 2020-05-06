#include "pch.h"

#include "RL_PPO.h"
#include "BitBotConstants.h"

// https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/PPO_continuous.py


RL_PPO_ReplayBuffer::RL_PPO_ReplayBuffer(void) :
    length(0)
{
    actions = torch::zeros({ BitSim::Trader::PPO::buffer_size, BitSim::Trader::action_dim_continuous });
    states = torch::zeros({ BitSim::Trader::PPO::buffer_size, BitSim::Trader::action_dim_continuous });
    log_probs = torch::zeros({ BitSim::Trader::PPO::buffer_size, 1 });
    rewards = torch::zeros({ BitSim::Trader::PPO::buffer_size, 1 });
    next_states = torch::zeros({ BitSim::Trader::PPO::buffer_size, BitSim::Trader::state_dim });
}

void RL_PPO_ReplayBuffer::clear(void)
{
    //length = 0;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> RL_PPO_ReplayBuffer::sample(void)
{
    return std::make_tuple(
        states, //.narrow(0, 0, length),
        actions, //.narrow(0, 0, length),
        log_probs, //.narrow(0, 0, length),
        rewards, //.narrow(0, 0, length),
        next_states //.narrow(0, 0, length)
    );
}

RL_PPO_ActorImpl::RL_PPO_ActorImpl(const std::string& name)
{
    actor->push_back(register_module(name + "_actor_linear_1", torch::nn::Linear{ BitSim::Trader::state_dim, BitSim::Trader::PPO::hidden_dim }));
    actor->push_back(register_module(name + "_actor_tanh_1", torch::nn::ReLU6{}));
    //actor->push_back(register_module(name + "_actor_linear_2", torch::nn::Linear{ BitSim::Trader::PPO::hidden_dim, BitSim::Trader::PPO::hidden_dim / 2 }));
    //actor->push_back(register_module(name + "_actor_tanh_2", torch::nn::ReLU6{}));

    actor_mean->push_back(register_module(name + "_actor_mean_linear_1", torch::nn::Linear{ BitSim::Trader::PPO::hidden_dim, BitSim::Trader::action_dim_continuous }));
    actor_mean->push_back(register_module(name + "_actor_mean_tanh_1", torch::nn::Tanh{}));

    actor_log_std->push_back(register_module(name + "_actor_log_std_linear_1", torch::nn::Linear{ BitSim::Trader::PPO::hidden_dim, BitSim::Trader::action_dim_continuous }));
    actor_log_std->push_back(register_module(name + "_actor_log_std_softplus_1", torch::nn::Softplus{}));
}

std::tuple<torch::Tensor, torch::Tensor> RL_PPO_ActorImpl::action(torch::Tensor state)
{
    auto action = torch::Tensor{};
    auto action_mean = torch::Tensor{};
    auto action_log_std = torch::Tensor{};
    auto action_std = torch::Tensor{};
    {
        const auto no_grad_guard = torch::NoGradGuard{};
        const auto latent = actor->forward(state);
        action_mean = 2.0 * actor_mean->forward(latent);
        action_log_std = actor_log_std->forward(latent);
        action_std = action_log_std.exp();
    }

    // Sample with reparametrization trick
    const auto eps = torch::normal(0.0, 1.0, action_std.sizes());
    action = action_mean + eps * action_std;

    auto log_prob = -(action - action_mean).pow(2) / (2 * action_std.pow(2)) - action_log_std - std::log(std::sqrt(2 * M_PI));
    //action = action.clamp(-BitSim::Trader::PPO::action_clamp, BitSim::Trader::PPO::action_clamp);

    return std::make_tuple(action, log_prob);
}

torch::Tensor RL_PPO_ActorImpl::log_prob(torch::Tensor state, torch::Tensor action)
{
    auto action_mean = torch::Tensor{};
    auto action_log_std = torch::Tensor{};
    auto action_std = torch::Tensor{};

    const auto latent = actor->forward(state);
    action_mean = 2.0 * actor_mean->forward(latent);
    action_log_std = actor_log_std->forward(latent);
    action_std = action_log_std.exp();

    auto log_prob = -(action - action_mean).pow(2) / (2 * action_std.pow(2)) - action_log_std - std::log(std::sqrt(2 * M_PI));
    return log_prob;
}

RL_PPO_CriticImpl::RL_PPO_CriticImpl(const std::string& name)
{
    critic->push_back(register_module(name + "_critic_linear_1", torch::nn::Linear{ BitSim::Trader::state_dim, BitSim::Trader::PPO::hidden_dim }));
    critic->push_back(register_module(name + "_critic_tanh_1", torch::nn::ReLU6{}));
    critic->push_back(register_module(name + "_critic_linear_2", torch::nn::Linear{ BitSim::Trader::PPO::hidden_dim, 1 }));
    //critic->push_back(register_module(name + "_critic_tanh_2", torch::nn::ReLU6{}));
    //critic->push_back(register_module(name + "_critic_linear_3", torch::nn::Linear{ BitSim::Trader::PPO::hidden_dim / 2, 1 }));
}

torch::Tensor RL_PPO_CriticImpl::forward(torch::Tensor state)
{
    auto state_value = critic->forward(state);
    return state_value;
}

RL_PPO::RL_PPO(void) :
    actor(RL_PPO_Actor{ "actor" }),
    critic(RL_PPO_Critic{ "critic" })
{
    optimizer_actor = std::make_unique<torch::optim::Adam>(actor->parameters(), BitSim::Trader::PPO::actor_learning_rate);
    optimizer_critic = std::make_unique<torch::optim::Adam>(critic->parameters(), BitSim::Trader::PPO::critic_learning_rate);
}

void RL_PPO::append_to_replay_buffer(sptrRL_State current_state, sptrRL_Action action, sptrRL_State next_state)
{
    replay_buffer.rewards[replay_buffer.length] = (next_state->reward + 8) / 8;
    replay_buffer.next_states[replay_buffer.length] = next_state->to_tensor()[0];
    replay_buffer.length = (replay_buffer.length + 1) % BitSim::Trader::PPO::buffer_size;
}

sptrRL_Action RL_PPO::get_action(sptrRL_State state)
{
    const auto [action, log_prob] = actor->action(state->to_tensor());

    replay_buffer.states[replay_buffer.length] = state->to_tensor()[0];
    replay_buffer.actions[replay_buffer.length] = action[0];
    replay_buffer.log_probs[replay_buffer.length] = log_prob[0];

    return std::make_shared<RL_Action>(action.view({ BitSim::Trader::action_dim_continuous }));
}

sptrRL_Action RL_PPO::get_random_action(sptrRL_State state)
{
    // Not used
    return std::make_shared<RL_Action>();
}

std::array<double, 6> RL_PPO::update_model(void)
{
    auto [states, actions, log_probs, rewards, next_states] = replay_buffer.sample();

    //static auto running_reward = 0.0;
    //running_reward = running_reward * 0.9 + rewards.sum().item().toDouble() * 0.1;
    const auto rewards_sum = rewards.sum().item().toDouble() / rewards.size(0);

    rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5);

    auto target_values = torch::Tensor{};
    {
        const auto no_grad_guard = torch::NoGradGuard{};
        target_values = rewards + BitSim::Trader::PPO::gamma_discount * critic->forward(next_states);
    }
    const auto advantages = (target_values - critic->forward(states)).detach();

    auto total_loss = 0.0;
    auto total_action_loss = 0.0;
    auto total_value_loss = 0.0;

    for (auto idx_epoch = 0; idx_epoch < BitSim::Trader::PPO::update_epochs; ++idx_epoch) {
        const auto indices = torch::randint(BitSim::Trader::SAC::buffer_size, BitSim::Trader::PPO::update_batch_size, torch::TensorOptions{}.dtype(torch::ScalarType::Long));

        const auto batch_actions = actions.index(indices).detach();
        const auto batch_states = states.index(indices).detach();
        const auto batch_log_probs = log_probs.index(indices).detach();
        const auto batch_advantages = advantages.index(indices).detach();
        const auto batch_target_values = target_values.index(indices).detach();
        
        const auto new_log_probs = actor->log_prob(batch_states, batch_actions);
        const auto ratio = (new_log_probs - batch_log_probs.detach()).exp();
        const auto surr1 = batch_advantages * ratio;
        const auto surr2 = batch_advantages * torch::clamp(ratio, 1.0 - BitSim::Trader::PPO::clip_param, 1.0 + BitSim::Trader::PPO::clip_param);
        
        const auto action_loss = -torch::min(surr1, surr2).mean();
        optimizer_actor->zero_grad();
        action_loss.backward();
        torch::nn::utils::clip_grad_norm_(actor->parameters(), BitSim::Trader::PPO::max_grad_norm);
        optimizer_actor->step();

        const auto value_loss = torch::smooth_l1_loss(critic->forward(batch_states), batch_target_values);
        optimizer_critic->zero_grad();
        value_loss.backward();
        torch::nn::utils::clip_grad_norm_(critic->parameters(), BitSim::Trader::PPO::max_grad_norm);
        optimizer_critic->step();

        total_action_loss += action_loss.item().toDouble();
        total_value_loss += value_loss.item().toDouble();
    }

    replay_buffer.clear();

    total_loss = total_action_loss + total_value_loss;
    return std::array<double, 6>{ total_loss, total_action_loss, total_value_loss, rewards_sum };
}


/*
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
*/

/*

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
*/
