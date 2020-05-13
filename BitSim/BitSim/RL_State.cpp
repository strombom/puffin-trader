#include "pch.h"
#include "RL_State.h"
#include "BitBotConstants.h"


RL_State::RL_State(
    double reward,
    double cart_x_position,
    double cart_x_velocity,
    double cart_y_position,
    double cart_y_velocity,
    double pole_angle, 
    double pole_velocity
) :
//RL_State::RL_State(double reward, double angle, double velocity) :
    //done(false), reward(reward), angle(angle), velocity(velocity)
    cart_x_position(cart_x_position), cart_x_velocity(cart_x_velocity),
    cart_y_position(cart_y_position), cart_y_velocity(cart_y_velocity),
    pole_angle(pole_angle), pole_velocity(pole_velocity),
    done(false), reward(0.0)
{

}

RL_State::RL_State(std::shared_ptr<RL_State> state) :
    //done(state->done), reward(state->reward), angle(state->angle), velocity(state->velocity)
    cart_x_position(state->cart_x_position), cart_x_velocity(state->cart_x_velocity),
    cart_y_position(state->cart_y_position), cart_y_velocity(state->cart_y_velocity),
    pole_angle(state->pole_angle), pole_velocity(state->pole_velocity),
    done(false), reward(0.0)
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
    return torch::tensor({ std::sin(pole_angle), std::cos(pole_angle), pole_velocity, cart_x_position, cart_x_velocity, cart_y_position, cart_y_velocity }).view({ 1, BitSim::Trader::state_dim });
    //return torch::tensor({ std::sin(angle), std::cos(angle), velocity }).view({ 1, BitSim::Trader::state_dim });
}
