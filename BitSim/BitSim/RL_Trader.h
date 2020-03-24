#pragma once
#include "pch.h"

#include "Logger.h"
#include "RL_State.h"
#include "RL_Networks.h"
#include "RL_Environment.h"
#include "RL_ReplayBuffer.h"
#include "FE_Observations.h"
#include "BitmexSimulator.h"


class RL_Trader
{
public:
    RL_Trader(torch::Tensor features, sptrBitmexSimulator simulator);

    void train(void);

private:
    torch::Tensor features;
    
    int step_total;
    int step_episode;

    RL_Networks networks;
    RL_Environment environment;
    RL_ReplayBuffer replay_buffer;
    RL_Action get_action(RL_State state);
    RL_State step(RL_State current_state, RL_Action action);
    void update_model(void);
    void save_params(int idx_period);
    void interim_test(void);

    CSVLogger csv_logger;
};
