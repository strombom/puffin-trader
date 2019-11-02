#include "DateTime.h"
#include "Logger.h"

DateTime::DateTime(void)
{
    time = boost::posix_time::microsec_clock::local_time();
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
    ss.imbue(std::locale(std::locale::classic(),
        new boost::posix_time::time_facet(format)));
    ss << time;
    return ss.str();
}
