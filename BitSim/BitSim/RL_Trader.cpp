#include "pch.h"

#include "RL_Trader.h"
#include "BitBotConstants.h"


RL_Trader::RL_Trader(sptrCartpoleSimulator simulator) :
    simulator(simulator),
    step_total(0),
    step_episode(0),
    csv_logger(BitSim::Trader::log_names, BitSim::Trader::log_path)
{
    if (BitSim::Trader::algorithm == "PPO") {
        rl_algorithm = std::make_unique<RL_PPO>();
    }
    else if (BitSim::Trader::algorithm == "SAC") {
        rl_algorithm = std::make_unique<RL_SAC>();
    }
}

void RL_Trader::train(void)
{
    for (auto idx_episode = 0; idx_episode < BitSim::Trader::n_episodes; ++idx_episode) {

        auto state = simulator->reset("cartpole_" + std::to_string(idx_episode) + ".csv");
        step_episode = 0;

        while (!state->is_done() && step_episode < BitSim::Trader::max_steps) {
            state = step(state);

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
    auto losses = rl_algorithm->update_model();
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

sptrRL_State RL_Trader::step(sptrRL_State state)
{
    auto action = sptrRL_Action{ nullptr };
    if (BitSim::Trader::algorithm == "SAC" && step_total < BitSim::Trader::initial_random_action) {
        action = rl_algorithm->get_random_action(state);
    }
    else {
        action = rl_algorithm->get_action(state);
    }

    const auto last_step = step_episode == BitSim::Trader::max_steps - 1;
    auto next_state = simulator->step(action, last_step);
    rl_algorithm->append_to_replay_buffer(state, action, next_state, next_state->done);
    return next_state;
}
