#include "pch.h"
#include "RL_State.h"
#include "BitBotConstants.h"


RL_State::RL_State(double reward, double cart_position, double cart_velocity, double pole_angle, double pole_velocity) :
    done(false), reward(reward), //angle(angle), velocity(velocity)
    cart_position(cart_position), cart_velocity(cart_velocity),
    pole_angle(pole_angle), pole_velocity(pole_velocity)
{

}

void RL_State::set_done(void)
{
    done = true;
}

bool RL_State::is_done(void) const
{
    return done;
}

torch::Tensor RL_State::to_tensor(void) const
{
    return torch::tensor({ std::sin(pole_angle), std::cos(pole_angle), pole_velocity, cart_position, cart_velocity }).view({ 1, BitSim::Trader::state_dim });
}
