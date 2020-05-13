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
    //+BitSim::Trader::action_dim_continuous + BitSim::Trader::action_dim_discrete
    auto linear_1 = torch::nn::Linear{ BitSim::Trader::state_dim + BitSim::Trader::action_dim_continuous, BitSim::Trader::SAC::hidden_dim };
    auto linear_2 = torch::nn::Linear{ BitSim::Trader::SAC::hidden_dim, BitSim::Trader::SAC::hidden_dim };
    auto linear_3 = torch::nn::Linear{ BitSim::Trader::SAC::hidden_dim, BitSim::Trader::SAC::hidden_dim };
    auto linear_4 = torch::nn::Linear{ BitSim::Trader::SAC::hidden_dim, BitSim::Trader::action_dim_continuous + BitSim::Trader::action_dim_discrete };

    layers->push_back(register_module(name + "_linear_1", linear_1));
    layers->push_back(register_module(name + "_relu_1", torch::nn::ReLU{}));
    layers->push_back(register_module(name + "_linear_2", linear_2));
    layers->push_back(register_module(name + "_relu_2", torch::nn::ReLU{}));
    layers->push_back(register_module(name + "_linear_3", linear_3));
    layers->push_back(register_module(name + "_relu_3", torch::nn::ReLU{}));
    layers->push_back(register_module(name + "_linear_4", linear_4));

    linear_1->apply(initialize_weights);
    linear_2->apply(initialize_weights);
    linear_3->apply(initialize_weights);
    linear_4->apply(initialize_weights);
}

torch::Tensor QNetworkImpl::forward(torch::Tensor state, torch::Tensor action)
{
    return layers->forward(torch::cat({ state, action }, 1)); //, 
}

PolicyNetworkImpl::PolicyNetworkImpl(const std::string& name)
{
    auto linear_1 = torch::nn::Linear{ BitSim::Trader::state_dim, BitSim::Trader::SAC::hidden_dim };
    auto linear_2 = torch::nn::Linear{ BitSim::Trader::SAC::hidden_dim, BitSim::Trader::SAC::hidden_dim };
    auto linear_3 = torch::nn::Linear{ BitSim::Trader::SAC::hidden_dim, BitSim::Trader::SAC::hidden_dim };
    auto linear_mean = torch::nn::Linear{ BitSim::Trader::SAC::hidden_dim, BitSim::Trader::action_dim_continuous };
    auto linear_log_std = torch::nn::Linear{ BitSim::Trader::SAC::hidden_dim, BitSim::Trader::action_dim_continuous };
    auto linear_discrete = torch::nn::Linear{ BitSim::Trader::SAC::hidden_dim, BitSim::Trader::action_dim_discrete };

    policy->push_back(register_module(name + "_linear_1", linear_1));
    policy->push_back(register_module(name + "_relu_1", torch::nn::ReLU{}));
    policy->push_back(register_module(name + "_linear_2", linear_2));
    policy->push_back(register_module(name + "_relu_2", torch::nn::ReLU{}));
    policy->push_back(register_module(name + "_linear_3", linear_3));
    policy->push_back(register_module(name + "_relu_3", torch::nn::ReLU{}));

    policy_mean->push_back(register_module(name + "_linear_mean", linear_mean));
    policy_log_std->push_back(register_module(name + "_linear_log_std", linear_log_std));
    policy_discrete->push_back(register_module(name + "_linear_discrete", linear_discrete));
    policy_discrete->push_back(register_module(name + "_softmax_discrete", torch::nn::Softmax{ 1 }));

    linear_1->apply(initialize_weights);
    linear_2->apply(initialize_weights);
    linear_3->apply(initialize_weights);
    linear_mean->apply(initialize_weights);
    linear_log_std->apply(initialize_weights);
    linear_discrete->apply(initialize_weights);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> PolicyNetworkImpl::forward(torch::Tensor state)
{
    const auto latent = policy->forward(state);
    const auto disc_action_prob = policy_discrete->forward(latent);
    const auto cont_mean = policy_mean->forward(latent);
    auto cont_log_std = policy_log_std->forward(latent);

    constexpr auto log_std_min = -20.0;
    constexpr auto log_std_max = 2.0;
    cont_log_std = torch::clamp(cont_log_std, log_std_min, log_std_max);

    return std::make_tuple(cont_mean, cont_log_std, disc_action_prob);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> PolicyNetworkImpl::sample_action(torch::Tensor state)
{
    // Discrete: https://medium.com/@kengz/soft-actor-critic-for-continuous-and-discrete-actions-eeff6f651954

    const auto action_scale = 2.0;

    const auto [mean, log_std, disc_prob] = forward(state);
    const auto std = log_std.exp();

    // Reparametrization trick
    const auto eps = torch::normal(0.0, 1.0, std.sizes()).to(BitSim::Trader::device);
    const auto z = mean + eps * std;
    const auto z_tanh = torch::tanh(z);
    const auto cont_action = action_scale * z_tanh;
    auto cont_log_prob = -(z - mean).pow(2) / (2 * std.pow(2)) - log_std - std::log(std::sqrt(2 * M_PI));
    auto cont_prob = torch::ones(cont_log_prob.sizes()).to(BitSim::Trader::device);

    // Enforce action bound
    if (BitSim::Trader::action_dim_continuous > 0) {
        cont_log_prob = cont_log_prob - (action_scale * (1 - z_tanh.pow(2)) + 1e-6).log();
        cont_log_prob = cont_log_prob.sum(1, true);
    }

    // Discrete actions
    auto disc_action_idx = torch::zeros({ state.size(0), 0 });
    auto disc_log_prob = torch::zeros({ state.size(0), 0 });
    if (BitSim::Trader::action_dim_discrete > 0) {
        disc_action_idx = disc_prob.multinomial(1);
        //auto disc_action = torch::zeros(disc_prob.sizes()).to(BitSim::Trader::device).scatter_(-1, disc_action_idx, 1);
        disc_log_prob = torch::log(disc_prob + 1e-8);
    }
    
    const auto action = cont_action; // torch::cat({ cont_action, disc_action_idx.toType(c10::ScalarType::Float) }, 1);
    const auto prob = torch::cat({ cont_prob, disc_prob }, 1);
    const auto log_prob = torch::cat({ cont_log_prob, disc_log_prob }, 1); ;

    return std::make_tuple(action, disc_action_idx, prob, log_prob);
}

RL_SAC_ReplayBuffer::RL_SAC_ReplayBuffer(void) :
    idx(0),
    length(0)
{
    current_states = torch::zeros({ BitSim::Trader::SAC::buffer_size, BitSim::Trader::state_dim }).to(BitSim::Trader::device);
    cont_actions = torch::zeros({ BitSim::Trader::SAC::buffer_size, BitSim::Trader::action_dim_continuous }).to(BitSim::Trader::device);
    disc_actions_idx = torch::zeros({ BitSim::Trader::SAC::buffer_size, 1 }, c10::TensorOptions{}.dtype(c10::ScalarType::Long)).to(BitSim::Trader::device);
    rewards = torch::zeros({ BitSim::Trader::SAC::buffer_size, 1 }).to(BitSim::Trader::device);
    next_states = torch::zeros({ BitSim::Trader::SAC::buffer_size, BitSim::Trader::state_dim }).to(BitSim::Trader::device);
    dones = torch::zeros({ BitSim::Trader::SAC::buffer_size, 1 }).to(BitSim::Trader::device);
}

void RL_SAC_ReplayBuffer::append(sptrRL_State current_state, sptrRL_Action action, sptrRL_State next_state)
{
    current_states[idx] = current_state->to_tensor().squeeze();
    cont_actions[idx] = action->to_tensor_cont();
    disc_actions_idx[idx] = action->to_tensor_disc();
    rewards[idx] = next_state->reward;
    next_states[idx] = next_state->to_tensor().squeeze();
    dones[idx] = next_state->done;

    idx = (idx + 1) % BitSim::Trader::SAC::buffer_size;
    length = std::min(length + 1, BitSim::Trader::SAC::buffer_size);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> RL_SAC_ReplayBuffer::sample(void)
{
    auto indices = torch::randint(length, BitSim::Trader::SAC::batch_size, torch::TensorOptions{}.dtype(torch::ScalarType::Long));
    //indices = (indices + BitSim::Trader::SAC::buffer_size + idx - length).fmod(BitSim::Trader::SAC::buffer_size);

    return std::make_tuple(
        current_states.index(indices).detach(),
        cont_actions.index(indices).detach(),
        disc_actions_idx.index(indices).detach(),
        rewards.index(indices).detach(),
        next_states.index(indices).detach(),
        dones.index(indices).detach()
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
    target_entropy(-BitSim::Trader::action_dim_continuous)
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
    const auto [action, disc_action_idx, _prob, _log_prob] = policy->sample_action(state_tensor);
    return std::make_shared<RL_Action>(action, disc_action_idx); // .view({ BitSim::Trader::action_dim_continuous + BitSim::Trader::action_dim_discrete }));
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
    auto [states, cont_actions, disc_actions_idx, rewards, next_states, dones] = replay_buffer.sample();

    // Critic
    auto q_target = torch::Tensor{};
    {
        auto no_grad_guard = torch::NoGradGuard{};
        const auto [next_cont_actions, _next_disc_actions_idx, next_probs, next_log_probs] = policy->sample_action(next_states);
        const auto next_q1_target = target_q1->forward(next_states, next_cont_actions);
        const auto next_q2_target = target_q2->forward(next_states, next_cont_actions);
        const auto next_q_target = (next_probs * (torch::min(next_q1_target, next_q2_target) - alpha * next_log_probs)).sum(1).unsqueeze(-1);
        q_target = rewards + BitSim::Trader::SAC::gamma_discount * next_q_target;
    }

    auto q1_pred = q1->forward(states, cont_actions);
    auto q2_pred = q2->forward(states, cont_actions);
    if (BitSim::Trader::action_dim_discrete > 0) {
        q1_pred = q1_pred.gather(1, disc_actions_idx);
        q2_pred = q2_pred.gather(1, disc_actions_idx);
    }
    const auto q1_loss = torch::mse_loss(q1_pred, q_target);
    const auto q2_loss = torch::mse_loss(q2_pred, q_target);
    //std::cout << std::endl << "---" << std::endl;
    //std::cout << q1_loss << std::endl;

    // Policy
    const auto [new_cont_actions, new_disc_actions_idx, new_probs, new_log_probs] = policy->sample_action(states);
    const auto new_q1_value = q1->forward(states, new_cont_actions);
    const auto new_q2_value = q2->forward(states, new_cont_actions);
    const auto new_q_value = torch::min(new_q1_value, new_q2_value);    
    const auto policy_loss = (new_probs * ((alpha * new_log_probs) - new_q_value)).mean();

    // Alpha
    const auto log_action_probabilities = (new_log_probs * new_probs).sum(1);
    const auto alpha_loss = -((log_action_probabilities + target_entropy).detach() * log_alpha).mean();

    // Optimize
    policy_optim->zero_grad();
    policy_loss.backward();
    policy_optim->step();

    q1_optim->zero_grad();
    q1_loss.backward();
    q1_optim->step();

    q2_optim->zero_grad();
    q2_loss.backward();
    q2_optim->step();

    alpha_optim->zero_grad();
    alpha_loss.backward();
    alpha_optim->step();

    // Update alpha
    alpha = log_alpha.exp().item().toDouble();

    // Soft update target
    auto param_q1 = q1->parameters();
    auto param_q2 = q1->parameters();
    auto target_param_q1 = target_q1->parameters();
    auto target_param_q2 = target_q2->parameters();
    for (auto i = 0; i < param_q1.size(); ++i) {
        target_param_q1[i].data().copy_(target_param_q1[i].data() * (1.0 - BitSim::Trader::SAC::soft_tau) + param_q1[i].data() * BitSim::Trader::SAC::soft_tau);
        target_param_q2[i].data().copy_(target_param_q2[i].data() * (1.0 - BitSim::Trader::SAC::soft_tau) + param_q2[i].data() * BitSim::Trader::SAC::soft_tau);
    }

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
