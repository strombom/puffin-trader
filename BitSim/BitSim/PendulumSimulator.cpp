#include "pch.h"
#include "Utils.h"
#include "PendulumSimulator.h"

// https://github.com/adik993/ppo-pytorch/blob/master/run_pendulum.py

PendulumSimulator::PendulumSimulator(void) :
    state(std::make_shared<RL_State>(0.0, 0.0, 0.0))
{

}

sptrRL_State PendulumSimulator::reset(int idx_episode)
{
    logger = std::make_unique<PendulumSimulatorLogger>("pendulum_" + std::to_string(idx_episode) + ".csv", true);

    state = std::make_shared<RL_State>(
        0.0,                                        // reward
        Utils::random(-M_PI, M_PI),                 // angle
        Utils::random(-max_speed, max_speed)        // velocity
        );
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

sptrRL_State PendulumSimulator::step(sptrRL_Action action, bool last_step)
{
    const auto torque = std::clamp(action->move, -max_torque, max_torque);

    const auto cost = std::pow(normalize_angle(state->angle), 2.0) + 0.1 * std::pow(state->velocity, 2.0) + 0.001 * std::pow(torque, 2.0);
    const auto new_velocity = state->velocity + (-3.0 * gravity / (2.0 * length) * std::sin(state->angle + M_PI) + 3.0 * torque / (mass * std::pow(length , 2.0))) * delta_time;
    const auto new_angle = state->angle + new_velocity * delta_time;

    state->velocity = std::clamp(new_velocity, -max_speed, max_speed);
    state->angle = new_angle;
    state->reward = -cost;

    logger->log(state->angle, state->velocity, state->reward);

    return state;
}

PendulumSimulatorLogger::PendulumSimulatorLogger(const std::string& filename, bool enabled) : enabled(enabled)
{
    if (enabled) {
        file.open(std::string{ BitSim::tmp_path } +"\\log\\" + filename);
        file << "angle,velocity,reward" << std::endl;
    }
}

void PendulumSimulatorLogger::log(double angle, double velocity, double reward)
{
    if (enabled) {
        file << angle << "," << velocity << "," << reward << std::endl;
    }
}
