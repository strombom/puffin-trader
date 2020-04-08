#include "pch.h"

#include "RL_Trader.h"
#include "BitBotConstants.h"


void RL_Trader::train(void)
{
    for (auto idx_episode = 0; idx_episode < BitSim::Trader::n_episodes; ++idx_episode) {

        auto state = simulator->reset("cartpole_" + std::to_string(idx_episode) + ".csv");
        step_episode = 0;

        while (!state.is_done() && step_episode < BitSim::Trader::max_steps) {
            const auto action = get_action(state);
            state = step(state, action);

            ++step_total;
            ++step_episode;
        }

        update_model(idx_episode);

        std::cout << "Steps: " << step_episode << std::endl;

        if (idx_episode % BitSim::Trader::save_period == 0 ||
            idx_episode == BitSim::Trader::n_episodes - 1) {
            save_params(idx_episode);
            interim_test();
        }
    }
}

void RL_Trader::update_model(double idx_episode)
{
    auto losses = networks.update_model();
    csv_logger.append_row(losses);

    std::cout << std::setfill(' ') << std::setw(4);
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Ep(" << idx_episode <<
        ") TL(" << losses[0] <<
        ") AcL(" << losses[1] <<
        ") AlL(" << losses[2] <<
        ") Q1(" << losses[3] <<
        ") Q2(" << losses[4] <<
        ") ES(" << losses[5] << ")" << std::endl;
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
    const auto last_step = step_episode == BitSim::Trader::max_steps - 1;
    auto next_state = simulator->step(action, last_step);
    networks.append_to_replay_buffer(current_state, action, next_state);
    return next_state;
}
