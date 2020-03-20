#include "pch.h"
#include "RL_Environment.h"


RL_State RL_Environment::reset(void)
{
    simulator->reset();

    constexpr auto reward = 0.0;
    return RL_State{ reward };
}

RL_State RL_Environment::step(const RL_Action& action)
{
    return simulator->step(action);
}
