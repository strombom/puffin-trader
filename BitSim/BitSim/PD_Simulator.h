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
    double get_mark_price(void);
    time_point_ms get_current_timestamp(void);
    double get_account_value(void);

    time_point_ms position_timestamp;
    double position_price;
    double position_direction;
    double position_stop_loss;

private:
    sptrPD_Events events;
    sptrES_Bitmex exchange;
    sptrAggTicks agg_ticks;

    time_point_ms training_start;
    time_point_ms training_end;
    time_point_ms validation_start;
    time_point_ms validation_end;

    size_t pd_events_idx;
    time_point_ms episode_end;

    double previous_value;

    torch::Tensor make_features(time_point_ms ref_timestamp, double ref_price);
    double calculate_reward(double mark_price);
};

using sptrPD_Simulator = std::shared_ptr<PD_Simulator>;
