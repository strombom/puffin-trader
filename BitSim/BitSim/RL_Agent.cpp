#include "pch.h"
#include "RL_Agent.h"
#include "BitBotConstants.h"


RL_Agent::RL_Agent(void) :
    agent_network(TanhGaussianDistParams{ "policy", BitSim::Trader::state_dim, BitSim::Trader::action_dim })
{

}
