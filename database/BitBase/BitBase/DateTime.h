#pragma once

#include "boost/date_time.hpp"

#include <string>

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

class TimeDelta
{
public:
    TimeDelta(const Duration& _duration);

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

class DateTime
{
public:
    DateTime(void); // Initialized with current date/time
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
