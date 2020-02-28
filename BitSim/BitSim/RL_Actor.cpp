#include "pch.h"

#include "RL_Actor.h"


RL_Action RL_Actor::get_action(RL_State state)
{
    // buy_position, buy_size, sell_position, sell_size
    return RL_Action{ 0.9, 0.5, 0.0, 0.0 };
}

RL_Action RL_Actor::get_random_action(void)
{
    return RL_Action::random();
}
