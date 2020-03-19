#pragma once
#include "pch.h"

#include "RL_State.h"
#include "RL_Action.h"
#include "RL_Networks.h"
#include "BitBotConstants.h"


class RL_Actor
{
public:
    RL_Actor(void) :
        policy(TanhGaussianDistParams{ "policy", BitSim::Trader::state_dim, BitSim::Trader::action_dim })
    {}

    RL_Action get_action(RL_State state);
    RL_Action get_random_action(void);

private:
    TanhGaussianDistParams policy;
    
};
