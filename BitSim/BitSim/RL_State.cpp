#include "pch.h"
#include "RL_State.h"
#include "BitBotConstants.h"


RL_State::RL_State(double reward, torch::Tensor features, double leverage) : 
    reward(reward), features(features), leverage(leverage), done(false)
{

}

RL_State::RL_State(std::shared_ptr<RL_State> state) :
    reward(0.0), features(state->features), leverage(state->leverage), done(false)
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
    return features.view({ 1, BitSim::Trader::state_dim });
    //return torch::cat({ features, torch::tensor({leverage}) }).view({ 1, BitSim::Trader::state_dim });

    //return torch::tensor({ std::sin(pole_ang), std::cos(pole_ang), pole_vel, cart_x_pos, cart_x_vel, cart_y_pos, cart_y_vel }).view({ 1, BitSim::Trader::state_dim });
    //return torch::tensor({ std::sin(angle), std::cos(angle), velocity }).view({ 1, BitSim::Trader::state_dim });
}

/*
// Cartpole

RL_State::RL_State(
    double reward,
    double cart_x_pos,
    double cart_x_vel,
    double cart_y_pos,
    double cart_y_vel,
    double pole_ang,
    double pole_vel
) :
//RL_State::RL_State(double reward, double angle, double velocity) :
    //done(false), reward(reward), angle(angle), velocity(velocity)
    cart_x_pos(cart_x_pos), cart_x_vel(cart_x_vel),
    cart_y_pos(cart_y_pos), cart_y_vel(cart_y_vel),
    pole_ang(pole_ang), pole_vel(pole_vel),
    done(false), reward(0.0)
{

}

RL_State::RL_State(std::shared_ptr<RL_State> state) :
    //done(state->done), reward(state->reward), angle(state->angle), velocity(state->velocity)
    cart_x_pos(state->cart_x_pos), cart_x_vel(state->cart_x_vel),
    cart_y_pos(state->cart_y_pos), cart_y_vel(state->cart_y_vel),
    pole_ang(state->pole_ang), pole_vel(state->pole_vel),
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
    return torch::tensor({ std::sin(pole_ang), std::cos(pole_ang), pole_vel, cart_x_pos, cart_x_vel, cart_y_pos, cart_y_vel }).view({ 1, BitSim::Trader::state_dim });
    //return torch::tensor({ std::sin(angle), std::cos(angle), velocity }).view({ 1, BitSim::Trader::state_dim });
}
*/
