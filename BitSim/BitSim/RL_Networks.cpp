#include "pch.h"
#include "RL_Networks.h"
#include "BitBotConstants.h"


MultilayerPerceptronImpl::MultilayerPerceptronImpl(const std::string& name, int input_size, int output_size)
{
    // Hidden layers
    auto next_size = input_size;
    for (auto idx = 0; idx < BitSim::Trader::hidden_count; ++idx) {
        auto hidden_layer = register_module(name + "_linear_" + std::to_string(idx), torch::nn::Linear{ next_size, BitSim::Trader::hidden_size });
        auto activation = register_module(name + "_activation_" + std::to_string(idx), torch::nn::ReLU6{});

        layers->push_back(hidden_layer);
        layers->push_back(activation);
        
        next_size = BitSim::Trader::hidden_size;
    }

    // Output layer
    auto output_layer = register_module(name + "_linear_output", torch::nn::Linear{ BitSim::Trader::hidden_size, output_size });
    //auto activation = register_module(name + "_output_activation", torch::nn::ReLU6{});

    //constexpr auto init_w = 3e-3;
    //torch::nn::init::uniform_(output_layer->weight, -init_w, init_w);
    //torch::nn::init::uniform_(output_layer->bias, -init_w, init_w);

    layers->push_back(output_layer);
    //layers->push_back(activation);
}

torch::Tensor MultilayerPerceptronImpl::forward(torch::Tensor x)
{
    std::cout << "layers: " << layers << std::endl;
    return layers->forward(x);
}

SoftQNetworkImpl::SoftQNetworkImpl(const std::string& name, int input_size, int action_size) :
    mlp(register_module(name, MultilayerPerceptron{ name + "_mlp", input_size + action_size, 1 }))
{

}

torch::Tensor SoftQNetworkImpl::forward(torch::Tensor state, torch::Tensor action)
{
    return mlp->forward(torch::cat({ state, action }, -1));
}

GaussianDistImpl::GaussianDistImpl(const std::string& name, int input_size, int output_size) : 
    mlp(register_module(name, MultilayerPerceptron{ name + "_mlp", input_size, output_size })),
    mean_layer(register_module(name + "_mean", torch::nn::Sequential{ torch::nn::Linear{input_size, output_size}, torch::nn::Tanh{} })),
    log_std_layer(register_module(name + "_std", torch::nn::Sequential{ torch::nn::Linear{input_size, output_size}, torch::nn::Tanh{} }))
{
    /*
    constexpr auto init_w = 3e-3;    
    torch::nn::init::uniform_(mean->weight, -init_w, init_w);
    torch::nn::init::uniform_(mean->bias, -init_w, init_w);
    torch::nn::init::uniform_(std->weight, -init_w, init_w);
    torch::nn::init::uniform_(std->bias, -init_w, init_w);
    */
}

std::tuple<torch::Tensor, torch::Tensor> GaussianDistImpl::get_dist_params(torch::Tensor x)
{
    const auto mean = mean_layer->forward(x);
    const auto log_std = log_std_layer->forward(x);

    constexpr auto log_std_min = -20.0;
    constexpr auto log_std_max = 2.0;
    const auto std = torch::clamp(log_std, log_std_min, log_std_max);
    //const auto std = torch::exp(log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1.0));

    return std::make_tuple(mean, std);
}

TanhGaussianDistParamsImpl::TanhGaussianDistParamsImpl(const std::string& name, int input_size, int output_size) :
    gaussian_dist(register_module(name, GaussianDist{ name + "_gauss", input_size, output_size }))
{

}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> TanhGaussianDistParamsImpl::forward(torch::Tensor x)
{
    const auto [mean, log_std] = gaussian_dist->get_dist_params(x);
    const auto std = log_std.exp();

    // Reparametrization trick
    const auto eps = torch::normal(0.0, 1.0, std.sizes());
    const auto z = mean + eps * std;

    // Normalize action and log_prob. Appendix C https://arxiv.org/pdf/1812.05905.pdf
    constexpr auto epsilon = 1e-6;
    const auto action = torch::tanh(z);
    const auto dist_log_prob = -(z - mean).pow(2) / (2 * std.pow(2)) - std.log() - std::log(std::sqrt(2 * M_PI));
    const auto log_prob = (dist_log_prob - (1 - action.pow(2) + epsilon).log()).sum(1, true);

    return std::make_tuple(action, log_prob, z, mean, std);
}

RL_Networks::RL_Networks(void) :
    actor(TanhGaussianDistParams{ "actor", BitSim::Trader::state_dim, BitSim::Trader::action_dim }),
    soft_q1(SoftQNetwork{ "soft_q1", BitSim::Trader::state_dim, BitSim::Trader::action_dim }),
    soft_q2(SoftQNetwork{ "soft_q2", BitSim::Trader::state_dim, BitSim::Trader::action_dim }),
    target_soft_q1(SoftQNetwork{ "target_soft_q1", BitSim::Trader::state_dim, BitSim::Trader::action_dim }),
    target_soft_q2(SoftQNetwork{ "target_soft_q2", BitSim::Trader::state_dim, BitSim::Trader::action_dim }),
    log_alpha(torch::zeros(1, torch::requires_grad())),
    target_entropy(-BitSim::Trader::action_dim),
    update_count(0)
{
    alpha_optim = std::make_unique<torch::optim::Adam>(std::vector{ log_alpha }, BitSim::Trader::learning_rate_entropy);
    soft_q1_optim = std::make_unique<torch::optim::Adam>(soft_q1->parameters(), BitSim::Trader::learning_rate_qf_1);
    soft_q2_optim = std::make_unique<torch::optim::Adam>(soft_q2->parameters(), BitSim::Trader::learning_rate_qf_2);
    actor_optim = std::make_unique<torch::optim::Adam>(actor->parameters(), BitSim::Trader::learning_rate_actor);
}

RL_Action RL_Networks::get_action(RL_State state)
{
    const auto state_tensor = state.to_tensor().view({ 1, BitSim::Trader::state_dim });
    const auto [action, log_prob, z, mean, std] = actor->forward(state_tensor);
    return RL_Action{ action.view({ BitSim::Trader::action_dim }) };
}

RL_Action RL_Networks::get_random_action(void)
{
    return RL_Action::random();
}

std::array<double, 6> RL_Networks::update_model(torch::Tensor states, torch::Tensor actions, torch::Tensor rewards, torch::Tensor next_states)
{
    const auto pred_q1 = soft_q1->forward(states, actions);
    const auto pred_q2 = soft_q2->forward(states, actions);
    const auto [new_actions, log_prob, z, mean, std] = actor->forward(states);
    const auto [new_next_actions, next_log_prob, next_z, next_mean, next_std] = actor->forward(next_states);
    const auto normalized_rewards = BitSim::Trader::reward_scale * (rewards - rewards.mean(0)) / (rewards.std(0) + 1e-6);

    // Tune entropy
    const auto alpha_loss = (-log_alpha * (log_prob - target_entropy).detach()).mean();
    alpha_optim->zero_grad();
    alpha_loss.backward();
    alpha_optim->step();
    const auto alpha = log_alpha.exp();

    // Train Q
    const auto target_q1_ = target_soft_q1->forward(next_states, new_next_actions);
    const auto target_q2 = target_soft_q2->forward(next_states, new_next_actions);
    const auto target_q_min = torch::min(target_q1_, target_q2);
    const auto target_q_value = normalized_rewards + BitSim::Trader::gamma_discount * target_q_min;
    const auto q1_value_loss = torch::mse_loss(pred_q1, target_q_value.detach());
    const auto q2_value_loss = torch::mse_loss(pred_q2, target_q_value.detach());

    soft_q1_optim->zero_grad();
    q1_value_loss.backward();
    soft_q1_optim->step();

    soft_q2_optim->zero_grad();
    q2_value_loss.backward();
    soft_q2_optim->step();

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

    const auto episode_score = rewards.sum().item().toDouble();

    return std::array<double, 6>{ total_loss, actor_loss_d, alpha_loss_d, q1_value_loss_d, q2_value_loss_d, episode_score };
}
