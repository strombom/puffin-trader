#include "pch.h"

#include "RL_Actor.h"


RL_Action RL_Actor::get_action(RL_State state)
{
    const auto [action, log_prob, z, mean, std] = policy->forward(state.to_tensor());
    return RL_Action{ action };
}

RL_Action RL_Actor::get_random_action(void)
{
    return RL_Action::random();
}
