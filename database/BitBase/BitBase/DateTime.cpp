#include "DateTime.h"
#include "Logger.h"


DateTime::DateTime(void)
{
    time = boost::posix_time::microsec_clock::universal_time();
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
