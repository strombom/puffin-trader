#pragma once
#include "pch.h"

#include "RL_State.h"
#include "RL_Action.h"
#include "BitmexSimulator.h"


class RL_Environment
{
public:
    RL_Environment(sptrBitmexSimulator simulator) :
        simulator(simulator) {}

    RL_State reset(void);
    RL_State step(const RL_Action &action);
    double get_reward(void);
    RL_Action random_action(void);

private:
    sptrBitmexSimulator simulator;

};
