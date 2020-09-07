#include "pch.h"
#include "PD_Simulator.h"


PD_Simulator::PD_Simulator(sptrIntervals intervals, torch::Tensor features)
{
    simulator = std::make_shared<BitmexSimulator>(intervals, features);
}

sptrRL_State PD_Simulator::reset(int idx_episode, bool validation, double training_progress)
{
    auto state = simulator->reset(idx_episode, validation, training_progress);

    auto tick = Tick{};
    events = std::make_shared<PD_Events>(tick);

    return state;
}

sptrRL_State PD_Simulator::step(sptrRL_Action action)
{
    auto state = simulator->step(action);

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
    return simulator->get_start_timestamp();
}
