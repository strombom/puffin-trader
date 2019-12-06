
#include "DateTime.h"

Timer::Timer(void)
{
    restart();
}

void Timer::restart(void)
{
    start_time_point = std::chrono::steady_clock::now();
}

std::chrono::microseconds Timer::elapsed(void)
{
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start_time_point);
}
