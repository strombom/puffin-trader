#include "pch.h"
#include "RL_State.h"


RL_State::RL_State(double reward, double cart_position, double cart_velocity, double pole_angle, double pole_velocity) : 
    done(false), reward(reward), cart_position(cart_position), cart_velocity(cart_velocity), pole_angle(pole_angle), pole_velocity(pole_velocity)
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

//void RL_State::set_state(torch::Tensor features, double leverage)
//{
//    state = torch::cat({ features, torch::tensor({ leverage }) });
//}

torch::Tensor RL_State::to_tensor(void) const
{
    return torch::tensor({ reward, cart_position, cart_velocity, pole_angle, pole_velocity }).view({ 1, 5 });
}

/*
RL_State::RL_State(double reward, torch::Tensor features, double leverage) : done(false), reward(reward)
{
    state = torch::cat({ features, torch::tensor({ leverage }) });
}

void RL_State::set_done(void)
{
    done = true;
}

bool RL_State::is_done(void) const
{
    return done;
}

void RL_State::set_state(torch::Tensor features, double leverage)
{
    state = torch::cat({ features, torch::tensor({ leverage }) });
}

double RL_State::get_reward(void) const
{
    return reward;
}

torch::Tensor RL_State::to_tensor(void) const
{
    return state;
}
*/
