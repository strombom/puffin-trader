#pragma once
#include "pch.h"

#include "FE_Observations.h"
#include "RL_Actor.h"
#include "RL_Environment.h"
#include "RL_State.h"
#include "RL_Action.h"


class RL_Closer
{
public:
    RL_Closer(sptrFE_Observations observations) :
        observations(observations),
        step_total(0),
        step_episode(0)
    {}

    void train(void);

private:
    sptrFE_Observations observations;
    
    int step_total;
    int step_episode;

    RL_Actor actor;
    RL_Environment environment;
    RL_Action get_action(RL_State state);
    std::tuple<RL_State, bool> step(RL_Action action);
    void update_model(void);
    void save_params(int idx_period);
    void interim_test(void);
};
