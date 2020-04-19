#pragma once
#include "pch.h"

#include "Logger.h"
#include "RL_SAC.h"
#include "RL_PPO.h"
#include "RL_State.h"
#include "RL_Algorithm.h"
#include "FE_Observations.h"
//#include "BitmexSimulator.h"
#include "PendulumSimulator.h"


class RL_Trader
{
public:
    RL_Trader(sptrPendulumSimulator simulator);

    void train(void);

private:    
    int step_total;
    int step_episode;

    uptrRL_Algorithm rl_algorithm;
    sptrPendulumSimulator simulator;
    sptrRL_State step(sptrRL_State state);
    void update_model(double idx_episode);
    void save_params(int idx_period);
    void interim_test(void);

    CSVLogger csv_logger;
};
