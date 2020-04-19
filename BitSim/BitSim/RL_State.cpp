#include "pch.h"
#include "RL_State.h"


RL_State::RL_State(double reward, double angle, double velocity) : 
    done(false), reward(reward), angle(angle), velocity(velocity)
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
    return torch::tensor({ reward, angle, velocity }).view({ 1, 3 });
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
