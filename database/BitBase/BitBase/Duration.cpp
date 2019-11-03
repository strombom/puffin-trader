#include "Duration.h"


Duration::Duration(void)
{

}

Duration::Duration(boost::gregorian::date_duration _date_duration, boost::posix_time::time_duration _time_duration)
{
    date_duration = _date_duration;
    time_duration = _time_duration;
}

boost::gregorian::date_duration Duration::get_date_duration(void) const
{
    return date_duration;
}

boost::posix_time::time_duration Duration::get_time_duration(void) const
{
    return time_duration;
}

Duration Duration::days(int _days)
{
    boost::gregorian::date_duration date_duration(_days);
    boost::posix_time::time_duration time_duration;

    return Duration(date_duration, time_duration);
}

Duration Duration::hours(int _hours)
{
    boost::gregorian::date_duration date_duration;
    boost::posix_time::time_duration time_duration(_hours, 0, 0, 0);

    return Duration(date_duration, time_duration);
}

Duration& Duration::operator+=(const Duration& duration)
{
    date_duration += duration.get_date_duration();
    time_duration += duration.get_time_duration();
    return *this;
}

