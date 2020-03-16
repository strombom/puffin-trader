#include "pch.h"
#include "RL_Environment.h"


RL_State RL_Environment::reset(void)
{
    simulator->reset();

    return RL_State{};
}

RL_State RL_Environment::step(const RL_Action& action)
{
    return simulator->step(action);
}

double RL_Environment::get_reward(void)
{
    return simulator->get_reward();
}
