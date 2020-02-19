#include "pch.h"

#include "DateTime.h"
#include "BitmexSimulator.h"


void BitmexSimulator::reset(void)
{
    current_timestamp = DateTime::random_timestamp(intervals->get_timestamp_start(), intervals->get_timestamp_end(), BitSim::interval);

    std::cout << DateTime::to_string(current_timestamp) << std::endl;

}
