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

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> RL_PPO_ActorCriticImpl::act(torch::Tensor state)
{
    const auto actor_latent = actor->forward(state);
    const auto state_value = critic->forward(state);
    const auto action_mean = actor_mean->forward(actor_latent);
    const auto action_log_std = actor_log_std->forward(actor_latent);
    const auto action_std = action_log_std.exp();

    // Sample with reparametrization trick
    const auto eps = torch::normal(0.0, 1.0, action_std.sizes());
    const auto action = action_mean + eps * action_std;

    // Log prob and entropy
    const auto log_prob = -(action - action_mean).pow(2) / (2 * action_std.pow(2)) - action_log_std - std::log(std::sqrt(2 * M_PI));
    const auto entropy = 0.5 + 0.5 * std::log(2 * M_PI) + torch::log(action_std);

    return std::make_tuple(state_value, action.detach(), log_prob, entropy);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> RL_PPO_ActorCriticImpl::evaluate(torch::Tensor state, torch::Tensor action)
{
    const auto actor_latent = actor->forward(state);
    const auto state_value = critic->forward(state);
    const auto action_mean = actor_mean->forward(actor_latent);
    const auto action_log_std = actor_log_std->forward(actor_latent);
    const auto action_std = action_log_std.exp();

    // Log prob and entropy
    const auto log_prob = -(action - action_mean).pow(2) / (2 * action_std.pow(2)) - action_log_std - std::log(std::sqrt(2 * M_PI));
    const auto entropy = 0.5 + 0.5 * std::log(2 * M_PI) + torch::log(action_std);

    return std::make_tuple(state_value, action, log_prob, entropy);
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
    //last_state = next_state;
}

sptrRL_Action RL_PPO::get_action(sptrRL_State state)
{
    const auto [_state_value, action, log_prob, entropy] = policy->act(state->to_tensor());

    replay_buffer.states[replay_buffer.length] = state->to_tensor()[0];
    replay_buffer.actions[replay_buffer.length] = action[0];
    replay_buffer.log_probs[replay_buffer.length] = log_prob[0];

    return std::make_shared<RL_Action>(action.squeeze()); // view({ BitSim::Trader::action_dim }));
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
    auto norm_rewards = torch::empty_like(rewards);
    auto discounted_reward = 0.0;
    const auto length = rewards.size(0);
    for (auto idx = length - 1; idx >= 0; --idx) {
        if (dones[idx].item().toBool()) {
            discounted_reward = 0.0;
        }
        discounted_reward = rewards[idx].item().toDouble() + BitSim::Trader::gamma_discount * discounted_reward;
        norm_rewards[idx] = discounted_reward;
    }
    norm_rewards = (norm_rewards - norm_rewards.mean()) / (norm_rewards.std() + 1e-5);

    for (auto idx_epoch = 0; idx_epoch < BitSim::Trader::ppo_update_epochs; ++idx_epoch) {
        auto indices = torch::randint(length, BitSim::Trader::ppo_batch_size, torch::TensorOptions{}.dtype(torch::ScalarType::Long));
        auto b_old_states = states.index(indices).detach();
        auto b_old_actions = actions.index(indices).detach();
        auto b_old_log_probs = log_probs.index(indices).detach();
        auto b_rewards = norm_rewards.index(indices).detach();

        policy->train();
        {
            auto auto_grad_guard = torch::AutoGradMode{ true };

            auto [state_values, actions, log_probs, entropies] = policy->evaluate(b_old_states, b_old_actions);

            auto ratios = torch::exp(log_probs - b_old_log_probs.detach());

            auto advantages = b_rewards - state_values.detach();
            auto surr1 = ratios * advantages;
            auto surr2 = torch::clamp(ratios, 1.0 - eps_clip, 1.0 + eps_clip) * advantages;
            auto loss = -torch::min(surr1, surr2) + 0.5 * torch::mse_loss(state_values, norm_rewards) - 0.01 * entropies;

            optimizer->zero_grad();
            loss.mean().backward();
            optimizer->step();
        }
    }

    auto policy_params = policy->parameters();
    auto policy_old_params = policy_old->parameters();
    for (auto i = 0; i < policy_params.size(); ++i) {
        policy_old_params[i].data().copy_(policy_params[i].data());
    }

    auto loss = 0.0;
    auto pg_loss = 0.0;
    auto value_loss = 0.0;
    auto entropy_mean = 0.0;
    auto approx_kl = 0.0;

    return std::array<double, 6>{ loss, pg_loss, value_loss, entropy_mean, approx_kl };
}

/*


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
*/

/*
auto loss = 0.0;
auto pg_loss = 0.0;
auto value_loss = 0.0;
auto entropy_mean = 0.0;
auto approx_kl = 0.0;
//std::cout << "actions: " << actions << std::endl;

const auto n_batches = BitSim::Trader::ppo_n_updates * BitSim::Trader::ppo_n_batches;

for (auto idx_batch = 0; idx_batch < n_batches; ++idx_batch) {
    auto indices = torch::randint(length, BitSim::Trader::ppo_batch_size, torch::TensorOptions{}.dtype(torch::ScalarType::Long));

    auto s_states = states.index(indices).detach();
    auto s_old_actions = actions.index(indices).detach();
    auto s_old_values = values.index(indices).detach();
    auto s_old_neglogprobs = neglogprobs.index(indices).detach();
    auto s_dones = dones.index(indices).detach();
    auto s_returns = returns.index(indices).detach();

    auto advantages = torch::empty_like(returns).detach();
    {
        auto no_grad = torch::NoGradGuard{};
        advantages = s_returns - s_old_values;
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8);
    }

    policy->train();
    {
        auto auto_grad = torch::AutoGradMode{ true };
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
}

loss /= n_batches;
pg_loss /= n_batches;
value_loss /= n_batches;
entropy_mean /= n_batches;
approx_kl /= n_batches;
*/
/*
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
*/
