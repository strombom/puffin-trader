#pragma once

#include "Duration.h"

class TimeDelta
{
public:
    TimeDelta(const Duration& _duration);

    static TimeDelta days(int _days);
    static TimeDelta hours(int _hours);

    template <class ... TDurations>
    TimeDelta(const Duration& _duration, const TDurations&... _durations);

    boost::gregorian::date_duration get_date_duration(void) const;
    boost::posix_time::time_duration get_time_duration(void) const;

private:
    Duration duration;

    void add_delta(const Duration& _duration);

    template <class ... TDurations>
    void add_delta(const Duration& duration, const TDurations&... _durations);
};

template <class ... TDurations>
inline
TimeDelta::TimeDelta(const Duration& _duration, const TDurations&... _durations)
{
    duration = _duration;
    add_delta(_durations...);
}

template <class ... TDurations>
inline
void TimeDelta::add_delta(const Duration& _duration, const TDurations&... _durations)
{
    add_delta(_duration);
    add_delta(_durations...);
}
