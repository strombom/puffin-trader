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
    sptrES_Bitmex exchange;
    sptrAggTicks agg_ticks;

    time_point_ms training_start;
    time_point_ms training_end;
    time_point_ms validation_start;
    time_point_ms validation_end;

    size_t agg_ticks_idx;
    time_point_ms episode_end;

    double orderbook_last_price;
    double time_since_leverage_change;

    sptrPD_Event find_next_event(void);
    torch::Tensor make_features(void);
};

using sptrPD_Simulator = std::shared_ptr<PD_Simulator>;
