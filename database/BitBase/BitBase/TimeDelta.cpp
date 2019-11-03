#include "TimeDelta.h"


TimeDelta::TimeDelta(const Duration& _duration)
{
    duration = _duration;
}

void TimeDelta::add_delta(const Duration& _duration)
{
    duration += _duration;
}

boost::gregorian::date_duration TimeDelta::get_date_duration(void) const
{
    return duration.get_date_duration();
}

boost::posix_time::time_duration TimeDelta::get_time_duration(void) const
{
    return duration.get_time_duration();
}

TimeDelta TimeDelta::days(int _days)
{
    return TimeDelta(Duration::days(_days));
}

TimeDelta TimeDelta::hours(int _hours)
{
    return TimeDelta(Duration::hours(_hours));
}
