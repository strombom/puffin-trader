#include "pch.h"
#include "PD_Simulator.h"


PD_Simulator::PD_Simulator(sptrIntervals intervals, torch::Tensor features)
{
    simulator = std::make_shared<BitmexSimulator>(intervals, features);
}

sptrRL_State PD_Simulator::reset(int idx_episode, bool validation, double training_progress)
{
    return nullptr;
}

sptrRL_State PD_Simulator::step(sptrRL_Action action)
{
    return nullptr;
}

time_point_ms PD_Simulator::get_start_timestamp(void)
{
    return simulator->get_start_timestamp();
}
