
#include "DateTime.h"


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

const time_point_us DateTime::to_time_point_us(const std::string& string)
{
    return to_time_point_us(string, time_format);
}

const time_point_us DateTime::to_time_point_us(const std::string& string, const std::string& time_format)
{
    auto value = std::istringstream{ string };
    auto time_point = std::chrono::time_point<std::chrono::system_clock, std::chrono::microseconds>{};// time_point_ms{};
    value >> date::parse(time_format, time_point);
    return std::chrono::time_point_cast<std::chrono::milliseconds, std::chrono::system_clock, std::chrono::microseconds>(time_point);
}

const time_point_us DateTime::iso8601_us_to_time_point_us(const std::string& string)
{
    auto value = std::istringstream{ string };
    auto time_point = std::chrono::time_point<std::chrono::system_clock, std::chrono::microseconds>{};
    value >> date::parse("%FT%TZ", time_point);
    return std::chrono::time_point_cast<std::chrono::milliseconds, std::chrono::system_clock, std::chrono::microseconds>(time_point);
}

const std::string DateTime::to_string(const time_point_us timestamp)
{
    return date::format(time_format, timestamp);
}

const std::string DateTime::to_string_iso_8601(const time_point_us timestamp)
{
    const auto string = date::format("%FT%TZ", timestamp);
    return string; // std::string{ string.begin(), string.end() };
}

const time_point_us DateTime::random_timestamp(const time_point_us timestamp_start, const time_point_us timestamp_end, const std::chrono::microseconds interval)
{
    static auto random_generator = std::mt19937{ std::random_device{}() };
    const auto start = timestamp_start.time_since_epoch().count();
    const auto end = timestamp_end.time_since_epoch().count();
    auto rand_int = std::uniform_int_distribution<long long>{ start, end }(random_generator);
    auto timestamp_int = rand_int - rand_int % interval.count();
    auto timestamp = time_point_us{ std::chrono::seconds{ timestamp_int } };
    return timestamp;
}

const time_point_us DateTime::now(void)
{
    return std::chrono::time_point_cast<std::chrono::milliseconds>(std::chrono::system_clock::now());
}
