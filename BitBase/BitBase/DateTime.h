#pragma once

#include "date.h"

using time_point_s = std::chrono::time_point<std::chrono::system_clock, std::chrono::seconds>;
using time_point_us = std::chrono::time_point<std::chrono::system_clock, std::chrono::microseconds>;

#define system_clock_us_now() std::chrono::time_point_cast<std::chrono::milliseconds>(std::chrono::system_clock::now())

class Timer {
public:
    Timer(void);

    void restart(void);
    std::chrono::microseconds elapsed(void);

private:
    std::chrono::steady_clock::time_point start_time_point;
};
