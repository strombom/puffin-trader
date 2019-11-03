#pragma once

#include <string>
#include "boost/date_time.hpp"

#include "TimeDelta.h"


class DateTime
{
public:
    DateTime(void); // Initialized with current UTC date/time
    DateTime(const std::string& string);
    DateTime(const boost::posix_time::ptime& _time);
    DateTime(int year, int month, int day, int hour, int minute, double second);

    std::string to_string(void) const;
    std::string to_string(const char* format) const;

    void set_hour(int hour);
    void set_minute(int minute);
    void set_second(double second);

    DateTime operator-(const TimeDelta& time_delta);

private:
    boost::posix_time::ptime time;
};
