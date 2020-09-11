#pragma once
#include "pch.h"

#include "RL_State.h"
#include "RL_Action.h"
#include "PD_Events.h"
#include "ES_Bitmex.h"
#include "BitLib/AggTicks.h"


class PD_Simulator
{
public:
    PD_Simulator(sptrAggTicks agg_ticks);

    sptrRL_State reset(int idx_episode, bool validation);
    sptrRL_State step(sptrRL_Action action);
    time_point_ms get_start_timestamp(void);

private:
    sptrPD_Events events;
    sptrES_Bitmex simulator;

    sptrPD_Event find_next_event(void);

    size_t training_start_idx;
    size_t training_end_idx;
    size_t validation_start_idx;
    size_t validation_end_idx;

    size_t agg_ticks_idx_start;
    size_t agg_ticks_idx_end;


};

using sptrPD_Simulator = std::shared_ptr<PD_Simulator>;
