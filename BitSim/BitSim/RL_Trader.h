#pragma once
#include "pch.h"

#include "BitLib/Logger.h"
#include "RL_SAC.h"
#include "RL_PPO.h"
#include "RL_State.h"
#include "RL_Algorithm.h"
#include "FE_Observations.h"
#include "PD_Simulator.h"
//#include "CartpoleSimulator.h"
//#include "PendulumSimulator.h"


class RL_Trader
{
public:
    RL_Trader(sptrPD_Simulator simulator);
    //RL_Trader(sptrCartpoleSimulator simulator);
    //RL_Trader(sptrPendulumSimulator simulator);

    void train(void);
    void evaluate(int idx_episode, time_point_ms start, time_point_ms end);

private:    
    int step_total;
    int step_episode;

    uptrRL_Algorithm rl_algorithm;
    //sptrCartpoleSimulator simulator;
    //sptrPendulumSimulator simulator;
    sptrPD_Simulator simulator;
    sptrRL_State step(sptrRL_State state, int max_steps);
    void update_model(int idx_episode);
    void save_params(int idx_period);
    void run_episode(int idx_episode, bool validation);

    CSVLogger csv_logger;
};
