#include "pch.h"
#include "Utils.h"
#include "CartpoleSimulator.h"

// https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py

CartpoleSimulator::CartpoleSimulator(void) : 
    state(std::make_shared<RL_State>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
{

}

sptrRL_State CartpoleSimulator::reset(int idx_episode)
{
    logger = std::make_unique<CartpoleSimulatorLogger>("cartpole_" + std::to_string(idx_episode) + ".csv", true);

    state = std::make_shared<RL_State>(
        0.0,                                        // reward
        Utils::random(-1.00, 1.00),                 // cart_x_position
        Utils::random(-0.05, 0.05),                 // cart_x_velocity
        Utils::random(-1.00, 1.00),                 // cart_y_position
        Utils::random(-0.05, 0.05),                 // cart_y_velocity
        //Utils::random(-0.05, 0.05),                 // pole_angle
        Utils::random(3.14 - 0.05, 3.14 + 0.05),    // pole_angle
        Utils::random(-0.05, 0.05));                // pole_velocity
    return state;
}

double normalize_angle(double angle)
{
    auto result = std::fmod(std::abs(angle) + 3.14, 2 * 3.14) - 3.14;;
    if (angle < 0) {
        result = -result;
    }
    return result;
}

sptrRL_State CartpoleSimulator::step(sptrRL_Action action, bool last_step)
{
    auto force_x = 0.0;
    auto force_y = 0.0;

    if (action->x_move_left) {
        force_x = -force_mag;
    }
    else if (action->x_move_right) {
        force_x = force_mag;
    }

    force_y = force_mag * std::clamp(action->y_move, -BitSim::Trader::PPO::action_clamp, BitSim::Trader::PPO::action_clamp) / BitSim::Trader::PPO::action_clamp;

    const auto cart_x_acc = force_x / total_mass;
    const auto cart_y_acc = force_y / total_mass;

    state->cart_x_position = 0.0; // state->cart_x_position + tau * state->cart_x_velocity;
    state->cart_x_velocity = 0.0; //state->cart_x_velocity + tau * cart_x_acc;
    state->cart_y_position = state->cart_y_position + tau * state->cart_y_velocity;
    state->cart_y_velocity = state->cart_y_velocity + tau * cart_y_acc;

    state->pole_angle = 0.0;
    state->pole_velocity = 0.0;

    auto cost = 0.0
        + 1.0 * std::pow(state->cart_x_position, 2.0)
        + 1.0 * std::pow(state->cart_y_position, 2.0);

    state->reward = -0.5 * cost;


    /*
    //*action->move_side; // std::clamp(action->move, -BitSim::Trader::PPO::action_clamp, BitSim::Trader::PPO::action_clamp);
    const auto costheta = std::cos(state->pole_angle);
    const auto sintheta = std::sin(state->pole_angle);

    //const auto temp_x = (force_x + polemass_length * state->pole_velocity * state->pole_velocity * sintheta) / total_mass;
    //const auto pole_acc = (gravity * sintheta - costheta * temp) / (length * (4.0 / 3.0 - mass_pole * costheta * costheta / total_mass));

    const auto temp_x = (force_x + polemass_length * state->pole_velocity * state->pole_velocity * sintheta) / total_mass;
    const auto temp_y = (force_y + polemass_length * state->pole_velocity * state->pole_velocity * costheta) / total_mass;
    const auto pole_acc = ((gravity + temp_y) * sintheta - costheta * temp_x) / (length * (4.0 / 3.0 - mass_pole * costheta * costheta / total_mass));

    const auto cart_x_acc = temp_x - polemass_length * pole_acc * costheta / total_mass;
    const auto cart_y_acc = temp_y - polemass_length * pole_acc * sintheta / total_mass;

    state->cart_x_position = state->cart_x_position + tau * state->cart_x_velocity;
    state->cart_x_velocity = state->cart_x_velocity + tau * cart_x_acc;
    state->cart_y_position = state->cart_y_position + tau * state->cart_y_velocity;
    state->cart_y_velocity = state->cart_y_velocity + tau * cart_y_acc;
    state->pole_angle = state->pole_angle + tau * state->pole_velocity;
    state->pole_velocity = state->pole_velocity + tau * pole_acc;
    
    auto out_of_bound = 0;
    if (state->cart_x_position < -x_threshold || state->cart_x_position > x_threshold ||
        state->pole_angle < -theta_threshold || state->pole_angle > theta_threshold) {
        out_of_bound = 1;
    }

    auto cost = 1.0 * std::pow(normalize_angle(state->pole_angle), 4.0)
        + 0.05 * std::pow(state->pole_velocity, 2.0)
        + 0.0001 * std::pow(force_x, 2.0)
        + 0.0001 * std::pow(force_y, 2.0)
        + 1.0 * std::pow(state->cart_x_position, 2.0)
        + 1.0 * std::pow(state->cart_y_position, 2.0); // +0.01 * std::pow(state->cart_velocity, 2.0);

    state->reward = 1.0 - 0.05 * cost;

    */



    //state->reward = cos(state->pole_angle) - 1.0 - 0.001 * std::pow(state->pole_velocity, 2.0) - 0.0001 * std::pow(force, 2.0) - 0.01 * std::pow(state->cart_position, 2.0);

    //state->reward = 0.0 - std::abs(state->cart_position) * 0.1 - std::abs(std::fmod(state->pole_angle, 3.14159)) * 1;
    //auto norm_angle = std::fmod(state->pole_angle + 2.0 * M_PI, M_PI);
    //state->reward = 1.0 - 0.01 * std::pow(norm_angle, 2.0) - 0.001 * std::pow(state->pole_velocity, 2.0) - 0.00001 * std::pow(force * 0.1, 2.0);

    logger->log(state->cart_x_position, state->cart_x_velocity, state->cart_y_position, state->cart_y_velocity, state->pole_angle, state->pole_velocity, state->reward);

    return state;
}

CartpoleSimulatorLogger::CartpoleSimulatorLogger(const std::string& log_filename, bool enabled) : enabled(enabled)
{
    if (enabled) {
        file.open(std::string{ BitSim::tmp_path } +"\\log\\" + log_filename);
        file << "cart_x_position,cart_x_velocity,cart_y_position,cart_y_velocity,pole_angle,pole_velocity,reward" << std::endl;
    }
}

void CartpoleSimulatorLogger::log(double cart_x_position, double cart_x_velocity, double cart_y_position, double cart_y_velocity, double pole_angle, double pole_velocity, double reward)
{
    if (enabled) {
        file << cart_x_position << "," << cart_x_velocity << "," << cart_y_position << "," << cart_y_velocity << "," << pole_angle << "," << pole_velocity << "," << reward << std::endl;
    }
}
