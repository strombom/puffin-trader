#include "pch.h"
#include "PD_Simulator.h"


PD_Simulator::PD_Simulator(sptrAggTicks agg_ticks)
{
    simulator = std::make_shared<ES_Bitmex>();

    // Find last valid agg tick
    const auto last_valid_timestamp = agg_ticks->rows.back().timestamp - BitSim::Trader::episode_length;
    last_valid_agg_ticks_start_idx = agg_ticks->rows.size() - 1;
    while (last_valid_agg_ticks_start_idx > 0) {
        if (agg_ticks->rows[last_valid_agg_ticks_start_idx].timestamp >= last_valid_timestamp) {
            last_valid_agg_ticks_start_idx--;
        }
        else {
            break;
        }
    }

}

sptrRL_State PD_Simulator::reset(int idx_episode, bool validation)
{
    //auto state = simulator->reset();

    //auto tick = Tick{};
    //events = std::make_shared<PD_Events>(tick);


    auto agg_ticks_idx_start = 0;
    auto agg_ticks_idx_end = 0;


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
