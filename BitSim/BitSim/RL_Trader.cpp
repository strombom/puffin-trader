#include "pch.h"

#include "RL_Trader.h"
#include "BitLib/BitBotConstants.h"


RL_Trader::RL_Trader(sptrPD_Simulator simulator) :
//RL_Trader::RL_Trader(sptrCartpoleSimulator simulator) :
//RL_Trader::RL_Trader(sptrPendulumSimulator simulator) :
    simulator(simulator),
    step_total(0),
    step_episode(0),
    loss_logger(BitSim::Trader::loss_log_names, BitSim::Trader::loss_log_path)
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
    //const auto training_progress = (float)idx_episode / BitSim::Trader::n_episodes;
    
    episode_logger = std::make_shared<CSVLogger>(BitSim::Trader::episode_log_names, std::string{BitSim::Trader::episode_log_path} + "_" + std::to_string(idx_episode) + ".csv");

    auto state = simulator->reset(idx_episode, validation);
    step_episode = 0;
    auto episode_reward = 0.0;

    /*
    auto all_features = torch::empty({ 100000, 14 });
    all_features[0] = state->features[0];
    auto feature_count = 1;
    while (!state->is_done()) {
        auto action = rl_algorithm->get_random_action(state);
        state = simulator->step(action);
        all_features[feature_count] = state->features[0];
        feature_count++;
    }
    std::cout << "feature_count " << feature_count << std::endl;
    std::cout << all_features[0] << std::endl;
    std::cout << all_features[1] << std::endl;
    std::cout << all_features[2] << std::endl;
    auto features_file = std::ofstream{ std::string{ BitSim::tmp_path } + "\\pd_events\\features.csv" };
    for (auto idx = 0; idx < feature_count; idx++) {
        for (auto col = 0; col < 14; col++) {
            features_file << all_features[idx][col].item().toDouble() << ",";
        }
        features_file << '\n';
    }
    features_file.close();
    */

    while (!state->is_done()) { // && step_episode < BitSim::Trader::max_steps) {
        if (!validation && BitSim::Trader::algorithm == "SAC" &&
            step_total % BitSim::Trader::SAC::update_interval == 0 &&
            step_total >= BitSim::Trader::SAC::batch_size) {
            update_model(idx_episode);
        }
        state = step(state, BitSim::Trader::max_steps);
        episode_reward += state->reward;

        auto log_state = std::array<double, 6>{};
        log_state[0] = simulator->get_mark_price();
        log_state[1] = (double)simulator->get_current_timestamp().time_since_epoch().count();
        log_state[2] = simulator->get_account_value();
        log_state[3] = simulator->position_price;
        log_state[4] = simulator->position_direction;
        log_state[5] = simulator->position_stop_loss;

        episode_logger->append_row(log_state);

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
    auto state = simulator->reset(idx_episode, true);
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
    loss_logger.append_row(losses);

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
