#include "pch.h"

#include "DateTime.h"
#include "BitmexSimulator.h"


void BitmexSimulator::reset(void)
{
    const auto timestamp_start = intervals->get_timestamp_start();
    const auto timestamp_end = intervals->get_timestamp_end() - BitSim::Closer::closing_timeout;

    current_timestamp = DateTime::random_timestamp(timestamp_start, timestamp_end, BitSim::interval);



    std::cout << DateTime::to_string(current_timestamp) << std::endl;

}
