#include "pch.h"

#include "DateTime.h"

#include <iostream>


Timer::Timer(void)
{
    restart();
}

void Timer::restart(void)
{
    start_time_point = std::chrono::steady_clock::now();
}

std::chrono::microseconds Timer::elapsed(void) const
{
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start_time_point);
}

void Timer::print_elapsed(const std::string& message) const
{
    std::cout << message << " elapsed: " << elapsed().count() / 1000.0f << "ms" << std::endl;
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
