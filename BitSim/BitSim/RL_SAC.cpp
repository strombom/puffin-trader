#include "pch.h"
#include "RL_SAC.h"
#include "BitBotConstants.h"

// https://github.com/ajaysub110/RLin200Lines


void initialize_weights(torch::nn::Module& module) {
    auto no_grad = torch::NoGradGuard{};
    auto p = module.named_parameters(false);
    auto w = p.find("weight");
    torch::nn::init::xavier_uniform_(*w);
    
} 

QNetworkImpl::QNetworkImpl(const std::string& name)
{
    auto linear_1 = torch::nn::Linear{ BitSim::Trader::state_dim + BitSim::Trader::action_dim, BitSim::Trader::SAC::hidden_dim };
    auto linear_2 = torch::nn::Linear{ BitSim::Trader::SAC::hidden_dim, BitSim::Trader::SAC::hidden_dim };
    auto linear_3 = torch::nn::Linear{ BitSim::Trader::SAC::hidden_dim, 1 };

    layers->push_back(register_module(name + "_linear_1", linear_1));
    layers->push_back(register_module(name + "_relu_1", torch::nn::ReLU{}));
    layers->push_back(register_module(name + "_linear_2", linear_2));
    layers->push_back(register_module(name + "_relu_2", torch::nn::ReLU{}));
    layers->push_back(register_module(name + "_linear_3", linear_3));

    linear_1->apply(initialize_weights);
    linear_2->apply(initialize_weights);
    linear_3->apply(initialize_weights);
}

torch::Tensor QNetworkImpl::forward(torch::Tensor state, torch::Tensor action)
{
    return layers->forward(torch::cat({ state, action }, 1));
}

PolicyNetworkImpl::PolicyNetworkImpl(const std::string& name)
{
    auto linear_1 = torch::nn::Linear{ BitSim::Trader::state_dim, BitSim::Trader::SAC::hidden_dim };
    auto linear_2 = torch::nn::Linear{ BitSim::Trader::SAC::hidden_dim, BitSim::Trader::SAC::hidden_dim };
    auto linear_mean = torch::nn::Linear{ BitSim::Trader::SAC::hidden_dim, BitSim::Trader::action_dim };
    auto linear_log_std = torch::nn::Linear{ BitSim::Trader::SAC::hidden_dim, BitSim::Trader::action_dim };

    policy->push_back(register_module(name + "_linear_1", linear_1));
    policy->push_back(register_module(name + "_relu_1", torch::nn::ReLU{}));
    policy->push_back(register_module(name + "_linear_2", linear_2));
    policy->push_back(register_module(name + "_relu_2", torch::nn::ReLU{}));

    policy_mean->push_back(register_module(name + "_linear_mean", linear_mean));
    policy_log_std->push_back(register_module(name + "_linear_log_std", linear_log_std));

    linear_1->apply(initialize_weights);
    linear_2->apply(initialize_weights);
    linear_mean->apply(initialize_weights);
    linear_log_std->apply(initialize_weights);
}

std::tuple<torch::Tensor, torch::Tensor> PolicyNetworkImpl::forward(torch::Tensor state)
{
    const auto latent = policy->forward(state);
    const auto mean = policy_mean->forward(latent);
    auto log_std = policy_log_std->forward(latent);

    constexpr auto log_std_min = -20.0;
    constexpr auto log_std_max = 2.0;
    log_std = torch::clamp(log_std, log_std_min, log_std_max);

    return std::make_tuple(mean, log_std);
}

std::tuple<torch::Tensor, torch::Tensor> PolicyNetworkImpl::sample_action(torch::Tensor state)
{
    const auto action_scale = 2.0;

    const auto [mean, log_std] = forward(state);
    const auto std = log_std.exp();

    // Reparametrization trick
    const auto eps = torch::normal(0.0, 1.0, std.sizes()).to(BitSim::Trader::device);
    const auto z = mean + eps * std;
    const auto z_tanh = torch::tanh(z);
    const auto action = action_scale * z_tanh;
    auto log_prob = -(z - mean).pow(2) / (2 * std.pow(2)) - log_std - std::log(std::sqrt(2 * M_PI));

    // Enforce action bound
    log_prob = log_prob - (action_scale * (1 - z_tanh.pow(2)) + 1e-6).log();
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
    rewards[idx] = next_state->reward;
    next_states[idx] = next_state->to_tensor().squeeze();
    dones[idx] = next_state->done;

    idx = (idx + 1) % BitSim::Trader::SAC::buffer_size;
    length = std::min(length + 1, BitSim::Trader::SAC::buffer_size);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> RL_SAC_ReplayBuffer::sample(void)
{
    auto indices = torch::randint(length, BitSim::Trader::SAC::batch_size, torch::TensorOptions{}.dtype(torch::ScalarType::Long));
    //indices = (indices + BitSim::Trader::SAC::buffer_size + idx - length).fmod(BitSim::Trader::SAC::buffer_size);

    return std::make_tuple(
        current_states.index(indices).detach().to(BitSim::Trader::device), 
        actions.index(indices).detach().to(BitSim::Trader::device),
        rewards.index(indices).detach().to(BitSim::Trader::device),
        next_states.index(indices).detach().to(BitSim::Trader::device),
        dones.index(indices).detach().to(BitSim::Trader::device)
    );
}

RL_SAC::RL_SAC(void) :
    policy(PolicyNetwork{ "policy" }),
    q1(QNetwork{ "q1" }),
    q2(QNetwork{ "q2" }),
    target_q1(QNetwork{ "target_q1" }),
    target_q2(QNetwork{ "target_q2" }),
    alpha(BitSim::Trader::SAC::alpha),
    log_alpha(torch::zeros(1)),
    target_entropy(-BitSim::Trader::action_dim)
{
    policy->to(BitSim::Trader::device);
    q1->to(BitSim::Trader::device);
    q2->to(BitSim::Trader::device);
    target_q1->to(BitSim::Trader::device);
    target_q2->to(BitSim::Trader::device);
    log_alpha = log_alpha.to(BitSim::Trader::device);
    log_alpha.set_requires_grad(true);

    policy_optim = std::make_unique<torch::optim::Adam>(policy->parameters(), BitSim::Trader::SAC::learning_rate_actor);
    q1_optim = std::make_unique<torch::optim::Adam>(q1->parameters(), BitSim::Trader::SAC::learning_rate_qf_1);
    q2_optim = std::make_unique<torch::optim::Adam>(q2->parameters(), BitSim::Trader::SAC::learning_rate_qf_2);
    alpha_optim = std::make_unique<torch::optim::Adam>(std::vector{ log_alpha }, BitSim::Trader::SAC::learning_rate_entropy);

    // Copy Q parameters
    auto param_q1 = q1->parameters();
    auto param_q2 = q1->parameters();
    auto target_param_q1 = target_q1->parameters();
    auto target_param_q2 = target_q2->parameters();
    for (auto i = 0; i < param_q1.size(); ++i) {
        target_param_q1[i].data().copy_(param_q1[i].data());
        target_param_q2[i].data().copy_(param_q2[i].data());
    }
}

sptrRL_Action RL_SAC::get_action(sptrRL_State state)
{
    const auto state_tensor = state->to_tensor().view({ 1, BitSim::Trader::state_dim }).to(BitSim::Trader::device);
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
    auto timer = Timer{};

    auto time_sample_1 = 0.0;
    auto time_sample_2 = 0.0;
    auto time_alpha = 0.0;
    auto time_q_target = 0.0;
    auto time_policy = 0.0;
    auto time_backward = 0.0;
    auto time_soft_update = 0.0;
    
    auto [states, actions, rewards, next_states, dones] = replay_buffer.sample();

    timer.restart();
    const auto [new_actions, new_log_probs] = policy->sample_action(states);
    time_sample_1 = (double) timer.elapsed().count() * 0.000001;

    timer.restart();
    const auto q1_pred = q1->forward(states, actions);
    const auto q2_pred = q2->forward(states, actions);
    time_sample_2 = (double) timer.elapsed().count() * 0.000001;

    timer.restart();
    const auto alpha_loss = -((new_log_probs + target_entropy).detach() * log_alpha).mean();
    alpha_optim->zero_grad();
    alpha_loss.backward();
    alpha_optim->step();
    alpha = log_alpha.exp().item().toDouble();
    time_alpha = (double) timer.elapsed().count() * 0.000001;

    timer.restart();
    auto q_target = torch::Tensor{};
    {
        auto no_grad_guard = torch::NoGradGuard{};
        const auto [next_actions, next_log_probs] = policy->sample_action(next_states);
        const auto next_target_q1 = target_q1->forward(next_states, next_actions);
        const auto next_target_q2 = target_q2->forward(next_states, next_actions);
        const auto next_target_q = torch::min(next_target_q1, next_target_q2) - alpha * next_log_probs;
        q_target = rewards + BitSim::Trader::SAC::gamma_discount * next_target_q;
    }
    time_q_target = (double) timer.elapsed().count() * 0.000001;

    timer.restart();
    const auto q1_loss = torch::mse_loss(q1_pred, q_target.detach());
    const auto q2_loss = torch::mse_loss(q2_pred, q_target.detach());

    const auto new_q1_value = q1->forward(states, new_actions);
    const auto new_q2_value = q2->forward(states, new_actions);
    const auto new_q_value = torch::min(new_q1_value, new_q2_value);
    const auto policy_loss = ((alpha * new_log_probs) - new_q_value).mean();
    time_policy = (double) timer.elapsed().count() * 0.000001;

    timer.restart();
    policy_optim->zero_grad();
    policy_loss.backward();
    policy_optim->step();

    q1_optim->zero_grad();
    q1_loss.backward();
    q1_optim->step();

    q2_optim->zero_grad();
    q2_loss.backward();
    q2_optim->step();
    time_backward = (double)timer.elapsed().count() * 0.000001;

    timer.restart();
    // Soft update target
    auto param_q1 = q1->parameters();
    auto param_q2 = q1->parameters();
    auto target_param_q1 = target_q1->parameters();
    auto target_param_q2 = target_q2->parameters();
    for (auto i = 0; i < param_q1.size(); ++i) {
        target_param_q1[i].data().copy_(target_param_q1[i].data() * (1.0 - BitSim::Trader::SAC::soft_tau) + param_q1[i].data() * BitSim::Trader::SAC::soft_tau);
        target_param_q2[i].data().copy_(target_param_q2[i].data() * (1.0 - BitSim::Trader::SAC::soft_tau) + param_q2[i].data() * BitSim::Trader::SAC::soft_tau);
    }
    time_soft_update = (double) timer.elapsed().count() * 0.000001;

    const auto episode_score = rewards.sum().item().toDouble() / BitSim::Trader::SAC::batch_size * BitSim::Trader::max_steps;

    const auto losses = std::array<double, 6>{
        q1_loss.item().toDouble(),
        q2_loss.item().toDouble(),
        policy_loss.item().toDouble(),
        alpha_loss.item().toDouble(),
        alpha,
        episode_score
    };
    /*
    std::cout << "ts1(" << time_sample_1 << ") ";
    std::cout << "ts2(" << time_sample_2 << ") ";
    std::cout << "tA(" << time_alpha << ") ";
    std::cout << "tQ(" << time_q_target << ") ";
    std::cout << "tP(" << time_policy << ") ";
    std::cout << "tB(" << time_backward << ") ";
    std::cout << "tSu(" << time_soft_update << ") ";
    std::cout << std::endl;
    */

    return losses;
}

void RL_SAC::save(const std::string& filename)
{

}

void RL_SAC::open(const std::string& filename)
{

}
