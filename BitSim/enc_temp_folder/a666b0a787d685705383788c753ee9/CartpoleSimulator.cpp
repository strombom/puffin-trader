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
        Utils::random(-2.00, 2.00),                 // cart_x_position
        Utils::random(-0.05, 0.05),                 // cart_x_velocity
        Utils::random(-2.00, 2.00),                 // cart_y_position
        Utils::random(-0.05, 0.05),                 // cart_y_velocity
        Utils::random(-0.05, 0.05),                 // pole_angle
        //Utils::random(3.14 - 0.5, 3.14 + 0.5),      // pole_angle
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

    force_y = gravity * (mass_cart + mass_pole) + force_mag * std::clamp(action->y_move, -BitSim::Trader::PPO::action_clamp, BitSim::Trader::PPO::action_clamp) / BitSim::Trader::PPO::action_clamp;

    auto t_ang = state->pole_ang;
    auto t_vel = state->pole_vel;
    auto x_pos = state->cart_x_pos;
    auto y_pos = state->cart_y_pos;

    const auto t_cos = std::cos(t_ang);
    const auto t_sin = std::sin(t_ang);
    
    const auto t_acc = (force_x * t_cos + force_y * t_sin) / (mass_pole * length * (t_sin * t_sin + t_cos * t_cos) - (mass_cart + mass_pole) * length);
    const auto x_acc = (force_x - mass_pole * length * (t_acc * t_cos - t_vel * t_vel * t_sin)) / (mass_cart + mass_pole);
    const auto y_acc = (force_y - gravity * (mass_cart + mass_pole) - mass_pole * length * (t_acc * t_sin + t_vel * t_vel * t_cos)) / (mass_cart + mass_pole);

    state->cart_x_vel = state->cart_x_vel + tau * x_acc;
    state->cart_x_pos = state->cart_x_pos + tau * state->cart_x_vel;
    state->cart_y_vel = state->cart_y_vel + tau * y_acc;
    state->cart_y_pos = state->cart_y_pos + tau * state->cart_y_vel;
    state->pole_vel = state->pole_vel + tau * t_acc;
    state->pole_ang = state->pole_ang + tau * state->pole_vel;

    auto cost = 1.0 * std::pow(normalize_angle(state->pole_ang + M_PI), 2.0)
        + 0.05 * std::pow(state->pole_vel, 2.0)
        + 0.0001 * std::pow(force_x, 2.0)
        + 0.0001 * std::pow(force_y, 2.0)
        + 1.0 * std::pow(state->cart_x_pos, 2.0)
        + 1.0 * std::pow(state->cart_y_pos, 2.0);

    state->reward = -0.1 * std::sqrt(cost);

    logger->log(state->cart_x_pos, state->cart_x_vel, state->cart_y_pos, state->cart_y_vel, state->pole_ang, state->pole_vel, state->reward);

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
