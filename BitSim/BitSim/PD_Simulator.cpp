#include "pch.h"
#include "PD_Simulator.h"


PD_Simulator::PD_Simulator(sptrAggTicks agg_ticks)
{
    simulator = std::make_shared<ES_Bitmex>();
}

sptrRL_State PD_Simulator::reset(int idx_episode, bool validation)
{
    //auto state = simulator->reset();

    //auto tick = Tick{};
    //events = std::make_shared<PD_Events>(tick);

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
