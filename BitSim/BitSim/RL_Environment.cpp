#include "pch.h"
#include "RL_Environment.h"


RL_State RL_Environment::reset(void)
{
    return simulator->reset();
}

RL_State RL_Environment::step(const RL_Action& action)
{
    return simulator->step(action);
}
