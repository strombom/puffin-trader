#pragma once
#include "pch.h"

#include "Intervals.h"


class BitmexSimulator
{
public:
    BitmexSimulator(sptrIntervals intervals) :
        intervals(intervals) {}

    void reset(void);

private:
    sptrIntervals intervals;
    time_point_s current_timestamp;
};

using sptrBitmexSimulator = std::shared_ptr<BitmexSimulator>;
