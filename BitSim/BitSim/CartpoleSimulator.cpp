#include "pch.h"
#include "Utils.h"
#include "CartpoleSimulator.h"


CartpoleSimulator::CartpoleSimulator(void) : 
    state(RL_State{ 0.0, 0.0, 0.0, 0.0, 0.0 })
{

}

RL_State CartpoleSimulator::reset(const std::string& log_filename)
{
    logger = std::make_unique<CartpoleSimulatorLogger>(log_filename, true);

    // cart_position, cart_velocity, pole_angle, pole_velocity
    state = RL_State{ 0.0, Utils::random(-0.05, 0.05), Utils::random(-0.05, 0.05), Utils::random(3.14-0.05, 3.14+0.05), Utils::random(-0.05, 0.05) };
    return state;
}

RL_State CartpoleSimulator::step(const RL_Action& action, bool last_step)
{
    const auto force = force_mag * action.move;
    const auto costheta = std::cos(state.pole_angle);
    const auto sintheta = std::sin(state.pole_angle);

    const auto temp = (force + polemass_length * state.pole_velocity * state.pole_velocity * sintheta) / total_mass;
    const auto pole_acc = (gravity * sintheta - costheta * temp) / (length * (4.0 / 3.0 - mass_pole * costheta * costheta / total_mass));
    const auto cart_acc = temp - polemass_length * pole_acc * costheta / total_mass;

    state.cart_position = state.cart_position + tau * state.cart_velocity;
    state.cart_velocity = state.cart_velocity + tau * cart_acc;
    state.pole_angle = state.pole_angle + tau * state.pole_velocity;
    state.pole_velocity = state.pole_velocity + tau * pole_acc;
    
    /*
    if (state.cart_position < -x_threshold || state.cart_position > x_threshold ||
        state.pole_angle < -theta_threshold || state.pole_angle > theta_threshold) {
        state.set_done();
        state.reward = 0.0;
    }
    else {
    */
        state.reward = 1.0 - std::abs(state.cart_position)*0.01 - std::abs(std::fmod(state.pole_angle, 3.14159))*0.1;
    //}


    logger->log(state.cart_position, state.cart_velocity, state.pole_angle, state.pole_velocity, state.reward);

    return state;
}

CartpoleSimulatorLogger::CartpoleSimulatorLogger(const std::string& filename, bool enabled) : enabled(enabled)
{
    if (enabled) {
        file.open(std::string{ BitSim::tmp_path } +"\\log\\" + filename);
        file << "cart_position,cart_velocity,pole_angle,pole_velocity" << std::endl;
    }
}

void CartpoleSimulatorLogger::log(double cart_position, double cart_velocity, double pole_angle, double pole_velocity, double reward)
{
    if (enabled) {
        file << cart_position << "," << cart_velocity << "," << pole_angle << "," << pole_velocity << "," << reward << std::endl;
    }
}
