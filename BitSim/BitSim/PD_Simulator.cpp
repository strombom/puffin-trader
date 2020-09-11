#include "pch.h"
#include "PD_Simulator.h"


PD_Simulator::PD_Simulator(sptrAggTicks agg_ticks)
{
    simulator = std::make_shared<ES_Bitmex>();

    const auto validation_end_timestamp = agg_ticks->rows.back().timestamp - BitSim::Trader::episode_length;
    const auto training_end_timestamp = agg_ticks->rows.front().timestamp + (validation_end_timestamp - agg_ticks->rows.front().timestamp) * 4 / 5;

    training_start_idx = 0;
    training_end_idx = 0;
    while (training_end_idx < agg_ticks->rows.size() && agg_ticks->rows[training_end_idx].timestamp < training_end_timestamp)
    {
        training_end_idx++;
    }
    validation_start_idx = training_end_idx + 1;
    validation_end_idx = validation_start_idx;
    while (validation_end_idx < agg_ticks->rows.size() && agg_ticks->rows[validation_end_idx].timestamp < validation_end_timestamp)
    {
        validation_end_idx++;
    }
}

sptrRL_State PD_Simulator::reset(int idx_episode, bool validation)
{
    //auto state = simulator->reset();

    //auto tick = Tick{};
    //events = std::make_shared<PD_Events>(tick);


    //agg_ticks_idx_start;
    //agg_ticks_idx_end;
    /*
    const auto training_start_idx = BitSim::FeatureEncoder::observation_length - 1;
    const auto validation_end_idx = (int)intervals->rows.size() - episode_length;

    const auto training_end_idx = (int)((validation_end_idx - training_start_idx) * 4.0 / 5.0) - 1;
    const auto validation_start_idx = training_end_idx + 1;
    */


    auto t = torch::Tensor{};
    auto state = std::make_shared<RL_State>(0.0, t, 0.0, 0.0, 0.0);

    return state;
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
