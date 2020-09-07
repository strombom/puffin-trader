#include "pch.h"

#include "RL_Trader.h"
#include "BitLib/BitBotConstants.h"


RL_Trader::RL_Trader(sptrPD_Simulator simulator) :
//RL_Trader::RL_Trader(sptrCartpoleSimulator simulator) :
//RL_Trader::RL_Trader(sptrPendulumSimulator simulator) :
    simulator(simulator),
    step_total(0),
    step_episode(0),
    csv_logger(BitSim::Trader::log_names, BitSim::Trader::log_path)
{
    if (BitSim::Trader::algorithm == "PPO") {
        //rl_algorithm = std::make_unique<RL_PPO>();
    }
    else if (BitSim::Trader::algorithm == "SAC") {
        rl_algorithm = std::make_unique<RL_SAC>();
    }
}

void RL_Trader::run_episode(int idx_episode, bool validation)
{
    const auto training_progress = (float)idx_episode / BitSim::Trader::n_episodes;

    auto state = simulator->reset(idx_episode, validation, training_progress);
    step_episode = 0;
    auto episode_reward = 0.0;

    while (!state->is_done()) { // && step_episode < BitSim::Trader::max_steps) {
        if (!validation && BitSim::Trader::algorithm == "SAC" &&
            step_total % BitSim::Trader::SAC::update_interval == 0 &&
            step_total >= BitSim::Trader::SAC::batch_size) {
            update_model(idx_episode);
        }
        state = step(state, BitSim::Trader::max_steps);
        episode_reward += state->reward;

        ++step_total;
        ++step_episode;
    }

    if (!validation && BitSim::Trader::algorithm == "PPO" && step_total >= BitSim::Trader::PPO::buffer_size) {
        update_model(idx_episode);
    }

    if (validation) {
        std::cout << "Val reward (" << DateTime::to_string(simulator->get_start_timestamp()) << "): " << episode_reward << " - "; // std::endl;
    } else {
        std::cout << DateTime::to_string(system_clock_ms_now()) << " Train reward: " << episode_reward << " - "; // std::endl;
    }
}

void RL_Trader::train(void)
{
    for (auto idx_episode = 0; idx_episode < BitSim::Trader::n_episodes; ++idx_episode) {

        // Training episode
        run_episode(idx_episode, false);

        // Validation episode
        run_episode(idx_episode, true);

        if (idx_episode == 0 ||
            idx_episode % BitSim::Trader::save_period == 0 ||
            idx_episode == BitSim::Trader::n_episodes - 1) {
            save_params(idx_episode);
        }
    }
}

void RL_Trader::evaluate(int idx_episode, time_point_ms start, time_point_ms end)
{
    auto state = simulator->reset(idx_episode, true, 1.0);
    step_episode = 0;
    auto episode_reward = 0.0;
    const auto max_steps = (int)(((const std::chrono::milliseconds)(end - start)).count() / 10000);

    while (!state->is_done()) {
        state = step(state, max_steps);
        episode_reward += state->reward;

        ++step_total;
        ++step_episode;
    }

    std::cout << "Val reward (" << DateTime::to_string(simulator->get_start_timestamp()) << "): " << episode_reward << " - "; // std::endl;
}

void RL_Trader::update_model(int idx_episode)
{
    auto losses = rl_algorithm->update_model();
    csv_logger.append_row(losses);

    std::cout << std::setfill(' ') << std::setw(5);
    std::cout << std::fixed << std::setprecision(4);

    static auto last_idx_episode = 0;

    if (BitSim::Trader::algorithm == "SAC" && last_idx_episode != idx_episode) {
        last_idx_episode = idx_episode;
        std::cout << "Ep(" << idx_episode - 1 <<
            ") Q1(" << losses[0] <<
            ") Q2(" << losses[1] <<
            ") PL(" << losses[2] <<
            ") AL(" << losses[3] <<
            ") A("  << losses[4] <<
            ") R("  << losses[5] << ")" << std::endl;
    }
    else if (BitSim::Trader::algorithm == "PPO") {
        std::cout << "Ep(" << idx_episode - 1 <<
            ") TL(" << losses[0] <<
            ") AL(" << losses[1] <<
            ") VL(" << losses[2] <<
            ") R("  << losses[3] << ")" << std::endl;
    }
}

void RL_Trader::save_params(int idx_period)
{
    constexpr auto path = "C:\\development\\github\\puffin-trader\\tmp\\rl";
    const auto name = std::string{ "model_" } + std::to_string(idx_period);

    rl_algorithm->save(path, name);
}

sptrRL_State RL_Trader::step(sptrRL_State state, int max_steps)
{
    auto action = sptrRL_Action{ nullptr };

    /*
    action = std::make_shared<RL_Action>();
    if (step_episode > 15 && step_episode < 35) {
        action->idle = false;
        action->limit_order = true;
        action->leverage = -5.0;
    }
    if (step_episode > 90 && step_episode < 110) {
        action->idle = false;
        action->limit_order = true;
        action->leverage = -10.0;
    }
    if (step_episode > 140 && step_episode < 160) {
        action->idle = false;
        action->limit_order = true;
        action->leverage = 0.0;
    }
    if (step_episode > 170 && step_episode < 190) {
        action->idle = false;
        action->limit_order = true;
        action->leverage = 5.0;
    }
    if (step_episode > 220 && step_episode < 240) {
        action->idle = false;
        action->limit_order = true;
        action->leverage = 10.0;
    }
    if (step_episode > 300 && step_episode < 320) {
        action->idle = false;
        action->limit_order = true;
        action->leverage = 0.0;
    }
    */

    if (BitSim::Trader::algorithm == "SAC" && step_total < BitSim::Trader::SAC::initial_random_action) {
        action = rl_algorithm->get_random_action(state);
    }
    else {
        auto no_grad_guard = torch::NoGradGuard{};
        action = rl_algorithm->get_action(state);
    }

    const auto last_state = std::make_shared<RL_State>( RL_State{ state } );
    auto next_state = simulator->step(action);
    rl_algorithm->append_to_replay_buffer(last_state, action, next_state);
    return next_state;
}
