#pragma once
#include "pch.h"

#include "Logger.h"
#include "RL_SAC.h"
#include "RL_PPO.h"
#include "RL_State.h"
#include "RL_Algorithm.h"
#include "FE_Observations.h"
#include "BitmexSimulator.h"
//#include "CartpoleSimulator.h"
//#include "PendulumSimulator.h"


class RL_Trader
{
public:
    RL_Trader(sptrBitmexSimulator simulator);
    //RL_Trader(sptrCartpoleSimulator simulator);
    //RL_Trader(sptrPendulumSimulator simulator);

    void train(void);

private:    
    int step_total;
    int step_episode;

    uptrRL_Algorithm rl_algorithm;
    //sptrCartpoleSimulator simulator;
    //sptrPendulumSimulator simulator;
    sptrBitmexSimulator simulator;
    sptrRL_State step(sptrRL_State state);
    void update_model(int idx_episode);
    void save_params(int idx_period);
    void run_episode(int idx_episode, bool validation);

    CSVLogger csv_logger;
};
