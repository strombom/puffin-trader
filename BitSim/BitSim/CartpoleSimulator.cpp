#include "pch.h"
#include "Utils.h"
#include "CartpoleSimulator.h"

// https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py

CartpoleSimulator::CartpoleSimulator(void) : 
    state(std::make_shared<RL_State>(0.0, 0.0, 0.0, 0.0, 0.0))
{

}

sptrRL_State CartpoleSimulator::reset(int idx_episode)
{
    logger = std::make_unique<CartpoleSimulatorLogger>("cartpole_" + std::to_string(idx_episode) + ".csv", true);

    state = std::make_shared<RL_State>(
        0.0,                                        // reward
        Utils::random(-0.05, 0.05),                 // cart_position
        Utils::random(-0.05, 0.05),                 // cart_velocity
        Utils::random(-0.05, 0.05),                 // pole_angle
        //Utils::random(3.14 - 0.05, 3.14 + 0.05),    // pole_angle
        Utils::random(-0.05, 0.05));                // pole_velocity
    return state;
}

sptrRL_State CartpoleSimulator::step(sptrRL_Action action, bool last_step)
{
    const auto force = force_mag * action->move; // std::clamp(action->move, -BitSim::Trader::PPO::action_clamp, BitSim::Trader::PPO::action_clamp);
    const auto costheta = std::cos(state->pole_angle);
    const auto sintheta = std::sin(state->pole_angle);

    const auto temp = (force + polemass_length * state->pole_velocity * state->pole_velocity * sintheta) / total_mass;
    const auto pole_acc = (gravity * sintheta - costheta * temp) / (length * (4.0 / 3.0 - mass_pole * costheta * costheta / total_mass));
    const auto cart_acc = temp - polemass_length * pole_acc * costheta / total_mass;

    state->cart_position = state->cart_position + tau * state->cart_velocity;
    state->cart_velocity = state->cart_velocity + tau * cart_acc;
    state->pole_angle = state->pole_angle + tau * state->pole_velocity;
    state->pole_velocity = state->pole_velocity + tau * pole_acc;
    
    /*
    if (state->cart_position < -x_threshold || state->cart_position > x_threshold ||
        state->pole_angle < -theta_threshold || state->pole_angle > theta_threshold) {
        state->set_done();
        state->reward = 0.0;
    }
    else {
        state->reward = 1.0;
    }

    */

    auto out_of_bound = 0;
    if (state->cart_position < -x_threshold || state->cart_position > x_threshold ||
        state->pole_angle < -theta_threshold || state->pole_angle > theta_threshold) {
        out_of_bound = 1;
    }

    const auto normalized_angle = std::fmod(state->pole_angle + M_PI, 2 * M_PI) - M_PI;

    auto cost = 1.0 * std::pow(normalized_angle, 2.0)
        + 0.5 * std::pow(state->pole_velocity, 2.0)
        + 0.0000 * std::pow(force, 2.0)
        + 0.1 * std::pow(state->cart_position, 2.0); // +0.01 * std::pow(state->cart_velocity, 2.0);


    state->reward = 0.0 - 0.2 * cost;



    //state->reward = cos(state->pole_angle) - 1.0 - 0.001 * std::pow(state->pole_velocity, 2.0) - 0.0001 * std::pow(force, 2.0) - 0.01 * std::pow(state->cart_position, 2.0);

    //state->reward = 0.0 - std::abs(state->cart_position) * 0.1 - std::abs(std::fmod(state->pole_angle, 3.14159)) * 1;
    //auto norm_angle = std::fmod(state->pole_angle + 2.0 * M_PI, M_PI);
    //state->reward = 1.0 - 0.01 * std::pow(norm_angle, 2.0) - 0.001 * std::pow(state->pole_velocity, 2.0) - 0.00001 * std::pow(force * 0.1, 2.0);

    logger->log(state->cart_position, state->cart_velocity, state->pole_angle, state->pole_velocity, state->reward);

    return state;
}

CartpoleSimulatorLogger::CartpoleSimulatorLogger(const std::string& log_filename, bool enabled) : enabled(enabled)
{
    if (enabled) {
        file.open(std::string{ BitSim::tmp_path } +"\\log\\" + log_filename);
        file << "cart_position,cart_velocity,pole_angle,pole_velocity,reward" << std::endl;
    }
}

void CartpoleSimulatorLogger::log(double cart_position, double cart_velocity, double pole_angle, double pole_velocity, double reward)
{
    if (enabled) {
        file << cart_position << "," << cart_velocity << "," << pole_angle << "," << pole_velocity << "," << reward << std::endl;
    }
}
