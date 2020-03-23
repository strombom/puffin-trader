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
    auto activation = register_module(name + "_output_activation", torch::nn::ReLU6{});

    //constexpr auto init_w = 3e-3;
    //torch::nn::init::uniform_(output_layer->weight, -init_w, init_w);
    //torch::nn::init::uniform_(output_layer->bias, -init_w, init_w);

    layers->push_back(output_layer);
    layers->push_back(activation);
}

torch::Tensor MultilayerPerceptronImpl::forward(torch::Tensor x)
{
    return layers->forward(x);
}

FlattenMultilayerPerceptronImpl::FlattenMultilayerPerceptronImpl(const std::string& name, int input_size, int output_size) :
    mlp(register_module(name, MultilayerPerceptron{ name + "_mlp", input_size, output_size }))
{

}

torch::Tensor FlattenMultilayerPerceptronImpl::forward(torch::Tensor x, torch::Tensor y)
{
    return mlp->forward(torch::cat({ x, y }, -1));
}

GaussianDistImpl::GaussianDistImpl(const std::string& name, int input_size, int output_size) : 
    mlp(register_module(name, MultilayerPerceptron{ name + "_mlp", input_size, output_size })),
    mean_layer(register_module(name + "_mean", torch::nn::Sequential{ torch::nn::Linear{input_size, output_size}, torch::nn::ReLU6{} })),
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
    x = mlp->forward(x);

    const auto mean = mean_layer->forward(x);
    const auto log_std = log_std_layer->forward(x);

    constexpr auto log_std_min = -20.0;
    constexpr auto log_std_max = 2.0;
    const auto std = torch::exp(log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1.0));

    return std::make_tuple(mean, std);
}

TanhGaussianDistParamsImpl::TanhGaussianDistParamsImpl(const std::string& name, int input_size, int output_size) :
    gaussian_dist(register_module(name, GaussianDist{ name + "_gauss", input_size, output_size }))
{

}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> TanhGaussianDistParamsImpl::forward(torch::Tensor x)
{
    const auto [mean, std] = gaussian_dist->get_dist_params(x);

    // Reparametrization trick
    const auto eps = torch::normal(0.0, 1.0, std.sizes());
    const auto z = mean + eps * std;

    // Normalize action and log_prob. Appendix C https://arxiv.org/pdf/1812.05905.pdf
    constexpr auto epsilon = 1e-6;
    const auto action = torch::tanh(z);
    const auto dist_log_prob = -(z - mean).pow(2) / (2 * std.pow(2)) - std.log() - std::log(std::sqrt(2 * M_PI));
    const auto log_prob = dist_log_prob - (1 - action.pow(2) + epsilon).log().sum(-1, true);

    return std::make_tuple(action, log_prob, z, mean, std);
}

RL_Networks::RL_Networks(void) :
    actor(TanhGaussianDistParams{ "actor", BitSim::Trader::state_dim, BitSim::Trader::action_dim }),
    vf(MultilayerPerceptron{ "vf", BitSim::Trader::state_dim, 1 }),
    vf_target(MultilayerPerceptron{ "vf_target", BitSim::Trader::state_dim, 1 }),
    qf_1(FlattenMultilayerPerceptron{ "qf_1", BitSim::Trader::state_dim + BitSim::Trader::action_dim, 1 }),
    qf_2(FlattenMultilayerPerceptron{ "vf", BitSim::Trader::state_dim + BitSim::Trader::action_dim, 1 }),
    log_alpha(torch::zeros(1)),
    alpha_optim(std::vector{ log_alpha }, BitSim::Trader::learning_rate_entropy),
    target_entropy(-BitSim::Trader::action_dim),
    qf_1_optim(qf_1->parameters(), BitSim::Trader::learning_rate_qf_1),
    qf_2_optim(qf_2->parameters(), BitSim::Trader::learning_rate_qf_2),
    vf_optim(vf->parameters(), BitSim::Trader::learning_rate_vf),
    actor_optim(actor->parameters(), BitSim::Trader::learning_rate_actor)
{

}

RL_Action RL_Networks::get_action(RL_State state)
{
    const auto [action, log_prob, z, mean, std] = actor->forward(state.to_tensor());
    return RL_Action{ action };
}

RL_Action RL_Networks::get_random_action(void)
{
    return RL_Action::random();
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> RL_Networks::forward_policy(torch::Tensor states)
{
    return actor->forward(states);
}

std::array<double, 5> RL_Networks::update_model(int step, torch::Tensor states, torch::Tensor actions, torch::Tensor rewards, torch::Tensor next_states)
{
    auto losses = std::array<double, 5>{};

    const auto [action, log_prob, z, mean, std] = forward_policy(states);

    // Tune entropy
    auto alpha_loss = (-log_alpha * (log_prob - target_entropy).detach()).mean();
    alpha_optim.zero_grad();
    alpha_loss.backward();
    alpha_optim.step();
    auto alpha = log_alpha.exp();

    // Q loss
    auto q_1_pred = qf_1->forward(states, actions);
    auto q_2_pred = qf_2->forward(states, actions);
    auto q_target = rewards + BitSim::Trader::gamma_discount * vf_target->forward(next_states);
    auto qf_1_loss = torch::mse_loss(q_1_pred, q_target.detach());
    auto qf_2_loss = torch::mse_loss(q_2_pred, q_target.detach());

    // V loss
    auto v_pred = vf->forward(states);
    auto q_pred = torch::min(qf_1->forward(states, next_states), qf_2->forward(states, next_states));
    auto v_target = q_pred - alpha * log_prob;
    auto vf_loss = torch::mse_loss(v_pred, v_target.detach());
    
    // Train Q
    qf_1_optim.zero_grad();
    qf_1_loss.backward();
    qf_1_optim.step();

    qf_2_optim.zero_grad();
    qf_2_loss.backward();
    qf_2_optim.step();

    // Train V
    vf_optim.zero_grad();
    vf_loss.backward();
    vf_optim.step();

    if (step % BitSim::Trader::policy_update_freq == 0) {
        auto advantage = q_pred - v_pred.detach();
        auto actor_loss = (alpha * log_prob - advantage).mean();
        losses[0] = actor_loss.item().toDouble();

        // Train actor
        actor_optim.zero_grad();
        actor_loss.backward();
        actor_optim.step();
    }

    losses[1] = qf_1_loss.item().toDouble();
    losses[2] = qf_2_loss.item().toDouble();
    losses[3] = vf_loss.item().toDouble();
    losses[4] = alpha_loss.item().toDouble();

    return losses;
}
