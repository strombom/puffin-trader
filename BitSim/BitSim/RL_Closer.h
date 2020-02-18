#pragma once
#include "pch.h"

#include "FE_Observations.h"


class RL_Closer
{
public:
    RL_Closer(sptrFE_Observations observations) :
        observations(observations),
        n_step_total(0)
    {}

    void train(void);

private:
    sptrFE_Observations observations;
    
    int n_step_total;

    class Action
    {

    };

    class State
    {

    };

    class Actor
    {
    public:
        Action get_action(State state);
    };

    class Environment
    {
    public:
        State reset(void);
        Action random_action(void);
    };

    Actor actor;
    Environment environment;
    Action get_action(State state);
};
