#pragma once

#include "boost/date_time.hpp"


class Duration
{
public:
    Duration(void);
    Duration(boost::gregorian::date_duration _date_duration, boost::posix_time::time_duration _time_duration);

    static Duration days(int _days);
    static Duration hours(int _hours);

    boost::gregorian::date_duration get_date_duration(void) const;
    boost::posix_time::time_duration get_time_duration(void) const;

    Duration& operator+=(const Duration& duration);

private:
    boost::gregorian::date_duration date_duration;
    boost::posix_time::time_duration time_duration;
};
