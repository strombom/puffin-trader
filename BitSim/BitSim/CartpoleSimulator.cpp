#include "pch.h"
#include "CartpoleSimulator.h"


CartpoleSimulatorLogger::CartpoleSimulatorLogger(const std::string& filename, bool enabled)
{

}


CartpoleSimulator::CartpoleSimulator(void) : 
    state(RL_State{ 0.0, 0.0, 0.0, 0.0, 0.0 })
{

}

RL_State CartpoleSimulator::reset(const std::string& log_filename)
{
    state = RL_State{ 0.0, 0.0, 0.0, 0.0, 0.0 };
    return state;
}

RL_State CartpoleSimulator::step(const RL_Action& action)
{
    const auto move = action.to_tensor()[0].item().toDouble();
    const auto move_left = move < 0.2;
    const auto move_right = move > 0.8;

    const auto force = move_left * force_mag - move_right * force_mag;
    const auto costheta = std::cos(state.pole_angle);
    const auto sintheta = std::sin(state.pole_angle);

    const auto temp = (force + polemass_length * state.pole_velocity * state.pole_velocity * sintheta) / total_mass;
    const auto pole_acc = (gravity * sintheta - costheta * temp) / (length * (4.0 / 3.0 - mass_pole * costheta * costheta / total_mass));
    const auto cart_acc = temp - polemass_length * pole_acc * costheta / total_mass;

    state.cart_position = state.cart_position + tau * state.cart_velocity;
    state.cart_velocity = state.cart_velocity + tau * cart_acc;
    state.pole_angle = state.pole_angle + tau * state.pole_velocity;
    state.pole_velocity = state.pole_velocity + tau * pole_acc;

    if (state.cart_position < -x_threshold || state.cart_position > x_threshold ||
        state.pole_angle < -theta_threshold || state.pole_angle > theta_threshold) {
        state.set_done();
    }

    if (!state.is_done()) {
        state.reward = 1.0;
    }
    else {
        state.reward = 0.0;
    }

    return state;
}
