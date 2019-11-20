#pragma once

#include <string>
#include "boost/date_time.hpp"

#include "TimeDelta.h"


class DateTime
{
public:
    DateTime(void);
    DateTime(std::uint64_t timestamp);
    DateTime(const std::string& string);
    DateTime(const std::string& string, const std::string& format);
    DateTime(const boost::posix_time::ptime& _time);
    DateTime(int year, int month, int day, int hour, int minute, double second);
    
    static DateTime now(void);

    std::string to_string(void) const;
    std::string to_string_date(void) const;
    std::string to_string_time(void) const;
    std::string to_string(const char* format) const;

    std::uint64_t to_timestamp(void) const;

    void set_time(int hour, int minute, double second);
    void set_hour(int hour);
    void set_minute(int minute);
    void set_second(double second);

    DateTime operator=(const TimeDelta& time_delta);
    DateTime operator-(const TimeDelta& time_delta);
    DateTime operator+(const TimeDelta& time_delta);
    DateTime& operator-=(const TimeDelta& time_delta);
    DateTime& operator+=(const TimeDelta& time_delta);

    bool operator<(const DateTime& date_time);
    bool operator>(const DateTime& date_time);
    bool operator<=(const DateTime& date_time);
    bool operator>=(const DateTime& date_time);

    static std::uint64_t string_to_timestamp(const std::string& string);
    static std::uint64_t string_to_timestamp(const std::string& string, const std::string& format);

private:
    boost::posix_time::ptime time;

};
