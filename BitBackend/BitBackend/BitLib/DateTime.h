#pragma once
#include "../precompiled_headers.h"


using namespace std::chrono_literals;

using time_point_us = std::chrono::time_point<std::chrono::system_clock, std::chrono::microseconds>;

#define system_clock_ms_now() std::chrono::time_point_cast<std::chrono::microseconds>(std::chrono::system_clock::now())


class Timer {
public:
    Timer(void);

    void restart(void);
    std::chrono::microseconds elapsed(void) const;
    void print_elapsed(const std::string& message) const;

private:
    std::chrono::steady_clock::time_point start_time_point;
};

class DateTime {
public:
    static const time_point_us to_time_point_us(const std::string& string);
    static const time_point_us to_time_point_us(const std::string& string, const std::string& time_format);
    static const time_point_us iso8601_us_to_time_point_us(const std::string& string);
    static const std::string to_string(const time_point_us);
    static const std::string to_string_iso_8601(const time_point_us);
    static const time_point_us random_timestamp(const time_point_us timestamp_start, const time_point_us timestamp_end, const std::chrono::microseconds interval);
    static const time_point_us now(void);

private:
    static constexpr auto time_format = "%F %T";
};
