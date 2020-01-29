#include "FE_Observations.h"

#include "BitBotConstants.h"


FE_Observations::FE_Observations(uptrIntervals intervals, time_point_s start_time) :
    start_time(start_time), interval(intervals->interval)
{

    auto n_observations = (long) intervals->rows.size();
    observations = torch::empty({ n_observations, 3, BitSim::observation_length });

}

void FE_Observations::save(const std::string& file_path)
{

}
