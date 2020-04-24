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

        auto state = simulator->reset(idx_episode);
        step_episode = 0;

        while (!state->is_done() && step_episode < BitSim::Trader::max_steps) {
            state = step(state);

            if (BitSim::Trader::algorithm == "SAC" && idx_episode % BitSim::Trader::SAC::update_interval == 0) {
                update_model(idx_episode);
            }

            ++step_total;
            ++step_episode;
        }

        if (BitSim::Trader::algorithm == "PPO" && step_total >= BitSim::Trader::PPO::buffer_size) {
            update_model(idx_episode);
        }

        //std::cout << "Steps: " << step_episode << std::endl;

        if (idx_episode % BitSim::Trader::save_period == 0 ||
            idx_episode == BitSim::Trader::n_episodes - 1) {
            save_params(idx_episode);
            interim_test();
        }
    }
}

void RL_Trader::update_model(int idx_episode)
{
    auto losses = rl_algorithm->update_model();
    csv_logger.append_row(losses);

    std::cout << std::setfill(' ') << std::setw(4);
    std::cout << std::fixed << std::setprecision(3);

    static auto last_idx_episode = 0;

    if (BitSim::Trader::algorithm == "SAC" && last_idx_episode != idx_episode) {
        last_idx_episode = idx_episode;
        std::cout << "Ep(" << idx_episode <<
            ") Q1(" << losses[0] <<
            ") Q2(" << losses[1] <<
            ") PL(" << losses[2] <<
            ") AL(" << losses[3] <<
            ") R(" << losses[4] << ")" << std::endl;
    }
    else if (BitSim::Trader::algorithm == "PPO") {
        std::cout << "Ep(" << idx_episode <<
            ") TL(" << losses[0] <<
            ") AL(" << losses[1] <<
            ") VL(" << losses[2] <<
            ") R("  << losses[3] << ")" << std::endl;
    }
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
    if (BitSim::Trader::algorithm == "SAC" && step_total < BitSim::Trader::SAC::initial_random_action) {
        action = rl_algorithm->get_random_action(state);
    }
    else {
        action = rl_algorithm->get_action(state);
    }

    const auto last_step = step_episode == BitSim::Trader::max_steps - 1;
    auto next_state = simulator->step(action, last_step);
    rl_algorithm->append_to_replay_buffer(state, action, next_state);
    return next_state;
}
