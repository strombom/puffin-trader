#include "pch.h"

#include "RL_Trader.h"
#include "BitBotConstants.h"


RL_Trader::RL_Trader(sptrBitmexSimulator simulator) :
    environment(RL_Environment{ simulator }),
    step_total(0),
    step_episode(0),
    csv_logger(BitSim::Trader::log_names, BitSim::Trader::log_path)
{

}

void RL_Trader::train(void)
{
    for (auto idx_episode = 0; idx_episode < BitSim::Trader::n_episodes; ++idx_episode) {
        auto state = environment.reset();
        step_episode = 0;

        while (!state.is_done()) {
            const auto action = get_action(state);
            state = step(state, action);

            ++step_total;
            ++step_episode;
        }

        update_model();

        if (idx_episode % BitSim::Trader::save_period == 0 ||
            idx_episode == BitSim::Trader::n_episodes - 1) {
            save_params(idx_episode);
            interim_test();
        }
    }
}

void RL_Trader::update_model(void)
{
    auto [states, actions, rewards, next_states] = replay_buffer.sample();

    auto losses = networks.update_model(step_total, states, actions, rewards, next_states);

    // Log
    csv_logger.append_row(losses);
}

void RL_Trader::save_params(int idx_period)
{

}

void RL_Trader::interim_test(void)
{

}

RL_Action RL_Trader::get_action(RL_State state)
{
    if (step_total < BitSim::Trader::initial_random_action) {
        return networks.get_random_action();
    }

    return networks.get_action(state);
}

RL_State RL_Trader::step(RL_State current_state, RL_Action action)
{
    auto next_state = environment.step(action);
    replay_buffer.append(current_state, action, next_state.get_reward(), next_state);
    return next_state;
}
