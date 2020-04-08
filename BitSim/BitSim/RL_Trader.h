#pragma once
#include "pch.h"

#include "Logger.h"
#include "RL_SAC.h"
#include "RL_State.h"
#include "FE_Observations.h"
//#include "BitmexSimulator.h"
#include "CartpoleSimulator.h"


class RL_Trader
{
public:
    RL_Trader(sptrCartpoleSimulator simulator) :
        simulator(simulator),
        step_total(0),
        step_episode(0),
        csv_logger(BitSim::Trader::log_names, BitSim::Trader::log_path) {}

    void train(void);

private:    
    int step_total;
    int step_episode;

    RL_SAC networks;
    sptrCartpoleSimulator simulator;
    RL_Action get_action(RL_State state);
    RL_State step(RL_State current_state, RL_Action action);
    void update_model(double idx_episode);
    void save_params(int idx_period);
    void interim_test(void);

    CSVLogger csv_logger;
};
