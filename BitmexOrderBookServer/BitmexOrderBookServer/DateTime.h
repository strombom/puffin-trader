#pragma once
#include "pch.h"


using time_point_ms = std::chrono::time_point<std::chrono::system_clock, std::chrono::milliseconds>;

#define system_clock_ms_now() std::chrono::time_point_cast<std::chrono::milliseconds>(std::chrono::system_clock::now())


template <class time_point>
static const std::string datetime_to_string(const time_point_ms timestamp)
{
    return date::format("%F %T", timestamp);
}

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
    static const time_point_ms to_time_point(const std::string& string);
    static const time_point_ms to_time_point(const std::string& string, const std::string& time_format);
    static const std::string to_string(const time_point_ms timestamp);
    static const std::string to_string_iso_8601(const time_point_ms timestamp);
    static const time_point_ms random_timestamp(const time_point_ms timestamp_start, const time_point_ms timestamp_end, const std::chrono::milliseconds interval);

private:
    static constexpr auto time_format = "%F %T";
};
