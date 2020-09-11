#include "pch.h"

#include "BitLib/Utils.h"
#include "PD_Simulator.h"


PD_Simulator::PD_Simulator(sptrAggTicks agg_ticks) :
    agg_ticks(agg_ticks)
{
    exchange = std::make_shared<ES_Bitmex>();

    training_start = agg_ticks->rows.front().timestamp;
    training_end = agg_ticks->rows.front().timestamp + (agg_ticks->rows.back().timestamp - agg_ticks->rows.front().timestamp) * 4 / 5;
    validation_start = training_end;
    validation_end = agg_ticks->rows.back().timestamp;
}

sptrRL_State PD_Simulator::reset(int idx_episode, bool validation)
{
    auto timestamp_start = agg_ticks->rows.front().timestamp;
    if (validation) {
        timestamp_start = Utils::random(validation_start, validation_end - BitSim::Trader::episode_length);
    }
    else {
        timestamp_start = Utils::random(training_start, training_end - BitSim::Trader::episode_length);
    }
    auto timestamp_end = timestamp_start + BitSim::Trader::episode_length;

    // Find start index
    agg_ticks_idx = 0;
    while (agg_ticks_idx < agg_ticks->rows.size() && agg_ticks->rows[agg_ticks_idx].timestamp < timestamp_start)
    {
        agg_ticks_idx++;
    }

    exchange->reset(agg_ticks->rows[agg_ticks_idx].low);

    const auto reward = 0.0;
    const auto features = make_features();
    const auto leverage = 0.0;
    const auto delta_price = 0.0;
    const auto time_since_change = 0.0;
    const auto state = std::make_shared<RL_State>(reward, features, leverage, delta_price, time_since_change);

    return state;
}

torch::Tensor PD_Simulator::make_features(void)
{
    auto features = torch::Tensor{};
    return features;
}

sptrRL_State PD_Simulator::step(sptrRL_Action action)
{
    //auto state = simulator->step(action);
    auto t = torch::Tensor{};
    auto state = std::make_shared<RL_State>(0.0, t, 0.0, 0.0, 0.0);

    return state;
}

sptrPD_Event PD_Simulator::find_next_event(void)
{
    auto tick = Tick{};
    auto event = events->step(tick);
    return event;
}

time_point_ms PD_Simulator::get_start_timestamp(void)
{
    return system_clock_ms_now(); // simulator->get_start_timestamp();
}
