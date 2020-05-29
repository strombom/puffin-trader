#include "pch.h"

#include "DateTime.h"

#include <iostream>
#include <random>


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
    std::cout << message << " elapsed: " << (float)elapsed().count() / 1000.0f << "ms" << std::endl;
}

const time_point_ms DateTime::to_time_point_ms(const std::string& string)
{
    return to_time_point_ms(string, time_format);
}

const time_point_ms DateTime::to_time_point_ms(const std::string& string, const std::string& time_format)
{
    auto value = std::istringstream{ string };
    auto time_point = time_point_ms{};
    value >> date::parse(time_format, time_point);
    return time_point;
}

const std::string DateTime::to_string(const time_point_ms timestamp)
{
    return date::format(time_format, timestamp);
}

const std::string DateTime::to_string_iso_8601(const time_point_ms timestamp)
{
    const auto string = date::format("%FT%TZ", timestamp);
    return std::string{string.begin(), string.end()};
}

const time_point_s DateTime::random_timestamp(const time_point_s timestamp_start, const time_point_s timestamp_end, const std::chrono::seconds interval)
{
    static auto random_generator = std::mt19937{ std::random_device{}() };
    const auto start = timestamp_start.time_since_epoch().count();
    const auto end = timestamp_end.time_since_epoch().count();
    auto rand_int = std::uniform_int_distribution<long long>{ start, end }(random_generator);
    auto timestamp_int = rand_int - rand_int % interval.count();
    auto timestamp = time_point_s{ std::chrono::seconds{ timestamp_int } };
    return timestamp;
}
