#include "pch.h"

#include "BitLib/Utils.h"
#include "PD_Simulator.h"


PD_Simulator::PD_Simulator(sptrAggTicks agg_ticks) :
    agg_ticks(agg_ticks)
{
    simulator = std::make_shared<ES_Bitmex>();

    training_start = agg_ticks->rows.front().timestamp;
    training_end = agg_ticks->rows.front().timestamp + (agg_ticks->rows.back().timestamp - agg_ticks->rows.front().timestamp) * 4 / 5;
    validation_start = training_end;
    validation_end = agg_ticks->rows.back().timestamp;
    

    /*
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
    */
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

    //std::cout << "ts " << DateTime::to_string(training_start_timestamp) << std::endl;
    //std::cout << "te " << DateTime::to_string(training_end_timestamp) << std::endl;
    //std::cout << "vs " << DateTime::to_string(validation_start_timestamp) << std::endl;
    //std::cout << "ve " << DateTime::to_string(validation_end_timestamp) << std::endl;


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
