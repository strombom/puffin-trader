#include "pch.h"

#include "RL_Actor.h"


RL_Action RL_Actor::get_action(RL_State state)
{
    return RL_Action{ 0.0, 0.0, 0.0, 0.0 };
}

RL_Action RL_Actor::get_random_action(void)
{
    return RL_Action::random();
}
