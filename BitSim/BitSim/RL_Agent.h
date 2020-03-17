#pragma once
#include "pch.h"

#include "RL_Networks.h"


class RL_Agent
{
public:
    RL_Agent(void);
    

private:
    TanhGaussianDistParams agent_network;
};
