#include "pch.h"
#include "RL_Agent.h"

RL_Agent::RL_Agent(void) :
    agent_network(TanhGaussianDistParams{"policy", 256, 256, 2})
{

}
