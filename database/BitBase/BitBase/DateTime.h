#pragma once

#include "boost/date_time.hpp"

#include <string>


class DateTime
{
public:
    DateTime(void); // Initialized with current date/time
    DateTime(int year, int month, int day, int hour, int minute, double second);

    std::string to_string(void) const;
    std::string to_string(const char* format) const;

private:
    boost::posix_time::ptime time;
};
