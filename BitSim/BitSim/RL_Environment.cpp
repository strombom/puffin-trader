#include "pch.h"
#include "RL_Environment.h"


RL_State RL_Environment::reset(void)
{
    simulator->reset();

    return RL_State{};
}

RL_Action RL_Environment::random_action(void)
{
    return RL_Action{};
}
