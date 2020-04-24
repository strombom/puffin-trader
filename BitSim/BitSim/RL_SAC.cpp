#include "pch.h"
#include "RL_SAC.h"
#include "BitBotConstants.h"

// https://github.com/ajaysub110/RLin200Lines


QNetworkImpl::QNetworkImpl(const std::string& name)
{
    layers->push_back(register_module(name + "_linear_1", torch::nn::Linear{ BitSim::Trader::state_dim + BitSim::Trader::action_dim, BitSim::Trader::SAC::hidden_dim }));
    layers->push_back(register_module(name + "_relu_1", torch::nn::ReLU6{}));
    layers->push_back(register_module(name + "_linear_2", torch::nn::Linear{ BitSim::Trader::SAC::hidden_dim, BitSim::Trader::SAC::hidden_dim }));
    layers->push_back(register_module(name + "_relu_2", torch::nn::ReLU6{}));
    layers->push_back(register_module(name + "_linear_3", torch::nn::Linear{ BitSim::Trader::SAC::hidden_dim, 1 }));
}

torch::Tensor QNetworkImpl::forward(torch::Tensor state, torch::Tensor action)
{
    return layers->forward(torch::cat({ state, action }, 1));
}

PolicyNetworkImpl::PolicyNetworkImpl(const std::string& name)
{
    policy->push_back(register_module(name + "_linear_1", torch::nn::Linear{ BitSim::Trader::state_dim + BitSim::Trader::action_dim, BitSim::Trader::SAC::hidden_dim }));
    policy->push_back(register_module(name + "_relu_1", torch::nn::ReLU6{}));
    policy->push_back(register_module(name + "_linear_2", torch::nn::Linear{ BitSim::Trader::SAC::hidden_dim, BitSim::Trader::SAC::hidden_dim }));
    policy->push_back(register_module(name + "_relu_2", torch::nn::ReLU6{}));

    policy_mean->push_back(register_module(name + "_linear_mean", torch::nn::Linear{ BitSim::Trader::SAC::hidden_dim, BitSim::Trader::action_dim }));
    policy_log_std->push_back(register_module(name + "_linear_log_std", torch::nn::Linear{ BitSim::Trader::SAC::hidden_dim, BitSim::Trader::action_dim }));
}

std::tuple<torch::Tensor, torch::Tensor> PolicyNetworkImpl::forward(torch::Tensor state)
{
    const auto latent = policy->forward(state);
    auto mean = policy_mean->forward(latent);
    auto log_std = policy_log_std->forward(latent);

    constexpr auto log_std_min = -20.0;
    constexpr auto log_std_max = 2.0;
    log_std = torch::clamp(log_std, log_std_min, log_std_max);

    return std::make_tuple(mean, log_std);
}

std::tuple<torch::Tensor, torch::Tensor> PolicyNetworkImpl::sample_action(torch::Tensor state)
{
    const auto [mean, log_std] = forward(state);
    const auto std = log_std.exp();

    // Reparametrization trick
    const auto eps = torch::normal(0.0, 1.0, std.sizes());
    const auto z = mean + eps * std;
    const auto action = torch::tanh(z);
    auto log_prob = -(z - mean).pow(2) / (2 * std.pow(2)) - log_std - std::log(std::sqrt(2 * M_PI));
    
    // Enforce action bound
    log_prob = -(1 - action.pow(2) + 1e-6).log();
    log_prob = log_prob.sum(1, true);

    return std::make_tuple(action, log_prob);
}

RL_SAC_ReplayBuffer::RL_SAC_ReplayBuffer(void) :
    idx(0),
    length(0)
{
    current_states = torch::zeros({ BitSim::Trader::SAC::buffer_size, BitSim::Trader::state_dim });
    actions = torch::zeros({ BitSim::Trader::SAC::buffer_size, BitSim::Trader::action_dim });
    rewards = torch::zeros({ BitSim::Trader::SAC::buffer_size, 1 });
    next_states = torch::zeros({ BitSim::Trader::SAC::buffer_size, BitSim::Trader::state_dim });
    dones = torch::zeros({ BitSim::Trader::SAC::buffer_size, 1 });
}

void RL_SAC_ReplayBuffer::append(sptrRL_State current_state, sptrRL_Action action, sptrRL_State next_state)
{
    current_states[idx] = current_state->to_tensor().squeeze();
    actions[idx] = action->to_tensor();
    rewards[idx] = current_state->reward;
    next_states[idx] = next_state->to_tensor().squeeze();
    dones[idx] = next_state->done;

    idx = (idx + 1) % BitSim::Trader::SAC::buffer_size;
    length = std::min(length + 1, BitSim::Trader::SAC::buffer_size);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> RL_SAC_ReplayBuffer::sample(void)
{
    auto indices = torch::randint(length, BitSim::Trader::SAC::batch_size, torch::TensorOptions{}.dtype(torch::ScalarType::Long));
    indices = (indices + BitSim::Trader::SAC::buffer_size + idx - length).fmod(BitSim::Trader::SAC::buffer_size);

    return std::make_tuple(current_states.index(indices), actions.index(indices), rewards.index(indices), next_states.index(indices), dones.index(indices));
}

RL_SAC::RL_SAC(void) :
    policy(PolicyNetwork{ "policy" }),
    q1(QNetwork{ "q1" }),
    q2(QNetwork{ "q1" }),
    target_q1(QNetwork{ "target_q1" }),
    target_q2(QNetwork{ "target_q2" }),
    alpha(BitSim::Trader::SAC::alpha * torch::ones(1)),
    log_alpha(torch::zeros(1, torch::requires_grad())),
    target_entropy(-BitSim::Trader::action_dim),
    update_count(0)
{
    policy_optim = std::make_unique<torch::optim::Adam>(policy->parameters(), BitSim::Trader::SAC::learning_rate_actor);
    q1_optim = std::make_unique<torch::optim::Adam>(q1->parameters(), BitSim::Trader::SAC::learning_rate_qf_1);
    q2_optim = std::make_unique<torch::optim::Adam>(q2->parameters(), BitSim::Trader::SAC::learning_rate_qf_2);
    alpha_optim = std::make_unique<torch::optim::Adam>(std::vector{ log_alpha }, BitSim::Trader::SAC::learning_rate_entropy);
}

sptrRL_Action RL_SAC::get_action(sptrRL_State state)
{
    const auto state_tensor = state->to_tensor().view({ 1, BitSim::Trader::state_dim });
    const auto [action, _log_prob] = policy->sample_action(state_tensor);
    return std::make_shared<RL_Action>(action.view({ BitSim::Trader::action_dim }));
}

sptrRL_Action RL_SAC::get_random_action(sptrRL_State state)
{
    return RL_Action::random();
}

void RL_SAC::append_to_replay_buffer(sptrRL_State current_state, sptrRL_Action action, sptrRL_State next_state)
{
    replay_buffer.append(current_state, action, next_state);
}

std::array<double, 6> RL_SAC::update_model(void)
{
    auto [states, actions, rewards, next_states, dones] = replay_buffer.sample();

    auto next_q = torch::Tensor{};
    {
        auto no_grad_guard = torch::NoGradGuard{};
        const auto [next_actions, next_log_probs] = policy->sample_action(next_states);
        const auto next_target_q1 = q1->forward(next_states, next_actions);
        const auto next_target_q2 = q2->forward(next_states, next_actions);
        const auto next_target_q = torch::min(next_target_q1, next_target_q2) - alpha * next_log_probs;
        next_q = rewards + BitSim::Trader::SAC::gamma_discount * next_target_q;
    }

    const auto q1_value = q1->forward(states, actions);
    const auto q2_value = q2->forward(states, actions);
    const auto q1_loss = torch::mse_loss(q1_value, next_q);
    const auto q2_loss = torch::mse_loss(q2_value, next_q);

    const auto [new_actions, new_log_probs] = policy->sample_action(states);
    const auto q1_policy = q1->forward(states, new_actions);
    const auto q2_policy = q2->forward(states, new_actions);
    const auto q_policy = torch::min(q1_policy, q2_policy);
    const auto policy_loss = ((alpha * new_log_probs) - q_policy).mean();

    const auto alpha_loss = -(log_alpha * (new_log_probs + target_entropy).detach()).mean();

    q1_optim->zero_grad();
    q1_loss.backward();
    q1_optim->step();

    q2_optim->zero_grad();
    q2_loss.backward();
    q2_optim->step();

    policy_optim->zero_grad();
    policy_loss.backward();
    policy_optim->step();

    alpha_optim->zero_grad();
    alpha_loss.backward();
    alpha_optim->step();

    alpha = log_alpha.exp();

    return std::array<double, 6>{ 0.0 };

    /*
    const auto [new_actions, log_prob, _z, _mean, _std] = actor->forward(states);
    const auto [new_next_actions, next_log_prob, _next_z, _next_mean, _next_std] = actor->forward(next_states);

    // Train Q
    const auto target_q1_ = target_soft_q1->forward(next_states, new_next_actions);
    const auto target_q2 = target_soft_q2->forward(next_states, new_next_actions);
    const auto target_q_min = torch::min(target_q1_, target_q2);
    const auto target_q_value = (rewards + BitSim::Trader::SAC::gamma_discount * target_q_min).detach();

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
        target_param_q1[i].data().copy_(target_param_q1[i].data() * (1.0 - BitSim::Trader::SAC::soft_tau) + param_q1[i].data() * BitSim::Trader::SAC::soft_tau);
        target_param_q2[i].data().copy_(target_param_q2[i].data() * (1.0 - BitSim::Trader::SAC::soft_tau) + param_q2[i].data() * BitSim::Trader::SAC::soft_tau);
    }

    const auto actor_loss_d = actor_loss.item().toDouble();
    const auto alpha_loss_d = alpha_loss.item().toDouble();
    const auto q1_value_loss_d = q1_value_loss.item().toDouble();
    const auto q2_value_loss_d = q2_value_loss.item().toDouble();
    const auto total_loss = actor_loss_d + alpha_loss_d + q1_value_loss_d + q2_value_loss_d;

    const auto episode_score = rewards.sum().item().toDouble() / BitSim::Trader::SAC::batch_size;

    return std::array<double, 6>{ total_loss, actor_loss_d, alpha_loss_d, q1_value_loss_d, q2_value_loss_d, episode_score };
    */
}

void RL_SAC::save(const std::string& filename)
{

}

void RL_SAC::open(const std::string& filename)
{

}
