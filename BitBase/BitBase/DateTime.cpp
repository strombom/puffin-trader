
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

const time_point_us DateTime::to_time_point_us(const std::string& string)
{
    auto value = std::istringstream{ string };
    auto time_point = time_point_us{};
    value >> date::parse(time_format, time_point);
    return time_point;
}

const std::string DateTime::to_string(const time_point_us timestamp)
{
    return date::format(time_format, timestamp);
}
