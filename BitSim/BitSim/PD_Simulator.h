#pragma once
#include "pch.h"

#include "RL_State.h"
#include "RL_Action.h"
#include "PD_Events.h"
#include "BitmexSimulator.h"
#include "BitLib/Intervals.h"


class PD_Simulator
{
public:
    PD_Simulator(sptrIntervals intervals, torch::Tensor features);

    sptrRL_State reset(int idx_episode, bool validation, double training_progress);
    sptrRL_State step(sptrRL_Action action);
    time_point_ms get_start_timestamp(void);

private:
    sptrPD_Events events;
    sptrBitmexSimulator simulator;

    sptrPD_Event find_next_event(void);
};

using sptrPD_Simulator = std::shared_ptr<PD_Simulator>;
