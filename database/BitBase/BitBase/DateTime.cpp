#include "DateTime.h"
#include "Logger.h"

DateTime::DateTime(void)
{
    time = boost::posix_time::microsec_clock::local_time();
}

DateTime::DateTime(const boost::posix_time::ptime& _time)
{
    time = boost::posix_time::ptime(_time);
}

DateTime::DateTime(const std::string& string)
{
    boost::posix_time::time_input_facet* tif = new boost::posix_time::time_input_facet("%Y-%m-%d %H:%M:%s");
    std::istringstream iss(string);
    iss.imbue(std::locale(std::locale::classic(), tif));
    iss >> time;
}

DateTime::DateTime(int year, int month, int day, int hour, int minute, double second)
{
    int microsecond = (int)(100000 * (second - (int)second));
    time = boost::posix_time::ptime(boost::gregorian::date(year, month, day), 
                                    boost::posix_time::hours(hour) + 
                                    boost::posix_time::minutes(minute) +
                                    boost::posix_time::seconds((int)second) +
                                    boost::posix_time::microseconds(microsecond));
}

std::string DateTime::to_string(void) const
{
    return to_string("%Y-%m-%d %H:%M:%s");
}

std::string DateTime::to_string(const char* format) const
{
    std::ostringstream ss;
    ss.imbue(std::locale(std::locale::classic(), new boost::posix_time::time_facet(format)));
    ss << time;
    return ss.str();
}

void DateTime::set_hour(int hour)
{
    time = boost::posix_time::ptime(time.date(),
            boost::posix_time::hours(hour) +
            boost::posix_time::minutes(time.time_of_day().minutes()) +
            boost::posix_time::seconds(time.time_of_day().seconds()) +
            boost::posix_time::microseconds(time.time_of_day().fractional_seconds()));
}

void DateTime::set_minute(int minute)
{
    time = boost::posix_time::ptime(time.date(),
        boost::posix_time::hours(time.time_of_day().hours()) +
        boost::posix_time::minutes(minute) +
        boost::posix_time::seconds(time.time_of_day().seconds()) +
        boost::posix_time::microseconds(time.time_of_day().fractional_seconds()));
}

void DateTime::set_second(double second)
{
    int microsecond = (int)(100000 * (second - (int)second));
    time = boost::posix_time::ptime(time.date(),
        boost::posix_time::hours(time.time_of_day().hours()) +
        boost::posix_time::minutes(time.time_of_day().minutes()) +
        boost::posix_time::seconds((int) second) +
        boost::posix_time::microseconds(microsecond));
}

DateTime DateTime::operator-(const TimeDelta& time_delta)
{
    return DateTime(time - time_delta.get_date_duration() - time_delta.get_time_duration());
}

inline
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
