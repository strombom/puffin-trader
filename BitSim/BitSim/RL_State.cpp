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

torch::Tensor RL_State::to_tensor(void) const
{
    return torch::tensor({ std::sin(angle), std::cos(angle), velocity }).view({ 1, 3 });
}
