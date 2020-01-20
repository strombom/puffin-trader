#pragma once

#include "Intervals.h"


class FE_Training
{
public:
    FE_Training(uptrIntervals intervals) :
        intervals(std::move(intervals)) {}

    void train(void);
    void test_learning_rate(void);
    void measure_observations(void);

private:
    uptrIntervals intervals;
};

