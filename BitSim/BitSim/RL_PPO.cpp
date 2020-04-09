#include "pch.h"

#include "RL_PPO.h"
#include "BitBotConstants.h"


RL_PPO_ActorCriticImpl::RL_PPO_ActorCriticImpl(const std::string& name)
{
    actor->push_back(register_module(name + "_actor_linear_1", torch::nn::Linear{ BitSim::Trader::state_dim, 64 }));
    actor->push_back(register_module(name + "_actor_tanh_1", torch::nn::Tanh{}));
    actor->push_back(register_module(name + "_actor_linear_2", torch::nn::Linear{ 64, 32 }));
    actor->push_back(register_module(name + "_actor_tanh_2", torch::nn::Tanh{}));
    actor->push_back(register_module(name + "_actor_linear_3", torch::nn::Linear{ 32, BitSim::Trader::action_dim }));
    actor->push_back(register_module(name + "_actor_tanh_3", torch::nn::Tanh{}));

    critic->push_back(register_module(name + "_critic_linear_1", torch::nn::Linear{ BitSim::Trader::state_dim, 64 }));
    critic->push_back(register_module(name + "_critic_tanh_1", torch::nn::Tanh{}));
    critic->push_back(register_module(name + "_critic_linear_2", torch::nn::Linear{ 64, 32 }));
    critic->push_back(register_module(name + "_critic_tanh_2", torch::nn::Tanh{}));
    critic->push_back(register_module(name + "_critic_linear_3", torch::nn::Linear{ 32, 1 }));
    critic->push_back(register_module(name + "_critic_tanh_3", torch::nn::Tanh{}));

    action_var = torch::full({ BitSim::Trader::action_dim }, BitSim::Trader::ppo_action_std);
}

torch::Tensor RL_PPO_ActorCriticImpl::act(torch::Tensor x)
{
    return actor->forward(x);
}

torch::Tensor RL_PPO_ActorCriticImpl::evaluate(torch::Tensor x)
{
    return critic->forward(x);
}

RL_PPO_ReplayBuffer::RL_PPO_ReplayBuffer(void) :
    length(0)
{
    current_states = torch::zeros({ BitSim::Trader::max_steps, BitSim::Trader::state_dim });
    actions = torch::zeros({ BitSim::Trader::max_steps, BitSim::Trader::action_dim });
    rewards = torch::zeros({ BitSim::Trader::max_steps });
    next_states = torch::zeros({ BitSim::Trader::max_steps, BitSim::Trader::state_dim });
    dones = torch::zeros({ BitSim::Trader::max_steps });
}

void RL_PPO_ReplayBuffer::append(const RL_State& current_state, const RL_Action& action, const RL_State& next_state, bool done)
{
    current_states[length] = current_state.to_tensor();
    actions[length] = action.to_tensor();
    rewards[length] = current_state.reward;
    next_states[length] = next_state.to_tensor();
    dones[length] = done;

    length++;
}

void RL_PPO_ReplayBuffer::clear(void)
{
    length = 0;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> RL_PPO_ReplayBuffer::sample(void)
{
    return std::make_tuple(
        current_states.narrow(0, 0, length), 
        actions.narrow(0, 0, length),
        rewards.narrow(0, 0, length),
        next_states.narrow(0, 0, length),
        dones.narrow(0, 0, length)
    );
}

RL_PPO::RL_PPO(void) :
    policy(RL_PPO_ActorCritic{ "policy" }),
    policy_old(RL_PPO_ActorCritic{ "policy_old" })
{

}

void RL_PPO::append_to_replay_buffer(const RL_State& current_state, const RL_Action& action, const RL_State& next_state, bool done)
{
    replay_buffer.append(current_state, action, next_state, done);
}

RL_Action RL_PPO::get_action(RL_State state)
{
    const auto action = policy_old->act(state.to_tensor());
    std::cout << "action " << action.view({ BitSim::Trader::action_dim }) << std::endl;

    return RL_Action{ action.view({ BitSim::Trader::action_dim }) };
    /*
    const auto state_tensor = state.to_tensor().view({ 1, BitSim::Trader::state_dim });
    const auto [action, log_prob, z, mean, std] = actor->forward(state_tensor);
    return RL_Action{ action.view({ BitSim::Trader::action_dim }) };
    */
}

RL_Action RL_PPO::get_random_action(void)
{
    return RL_Action::random();
}

std::array<double, 6> RL_PPO::update_model(void)
{
    auto [states, actions, rewards, next_states, dones] = replay_buffer.sample();
    replay_buffer.clear();

    //auto discounted_rewards = rewards + (BitSim::Trader::gamma_discount);

    std::cout << "rewards " << rewards << std::endl;

    auto discounted_reward = 0.0;
    auto discounted_rewards = torch::zeros({ rewards.size(0) });

    //for (auto idx = 0; idx < rewards.size(0); ++idx) {

    //}


    return std::array<double, 6>{ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };

    /*
    # Monte Carlo estimate of rewards :
    rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)) :
            if is_terminal :
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

    # Normalizing the rewards :
    rewards = torch.tensor(rewards).to(device)
    rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
    */

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
