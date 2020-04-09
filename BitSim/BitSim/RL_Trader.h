#pragma once
#include "pch.h"

#include "Logger.h"
#include "RL_SAC.h"
#include "RL_PPO.h"
#include "RL_State.h"
#include "RL_Algorithm.h"
#include "FE_Observations.h"
//#include "BitmexSimulator.h"
#include "CartpoleSimulator.h"


class RL_Trader
{
public:
    RL_Trader(sptrCartpoleSimulator simulator);

    void train(void);

private:    
    int step_total;
    int step_episode;

    uptrRL_Algorithm rl_algorithm;
    sptrCartpoleSimulator simulator;
    RL_Action get_action(RL_State state);
    RL_State step(RL_State current_state, RL_Action action);
    void update_model(double idx_episode);
    void save_params(int idx_period);
    void interim_test(void);

    CSVLogger csv_logger;
};
