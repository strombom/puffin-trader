#include "Logger.h"

#include <stdarg.h>

Logger logger;

Logger::Logger(void)
{
    time_facet = new boost::posix_time::time_facet("%Y-%m-%d %H:%M:%s");
}

std::string Logger::get_timestamp(void)
{
    std::ostringstream ss;
    ss.imbue(std::locale(std::locale::classic(), time_facet));
    ss << boost::posix_time::microsec_clock::local_time();
    return ss.str();
}

void Logger::info(const char* format, ...)
{
    printf("%s INFO: ", get_timestamp().c_str());
    va_list args;
    va_start(args, format);
    vprintf(format, args);
    va_end(args);
    printf("\n");
}

void Logger::warn(const char* format, ...)
{
    printf("%s WARN: ", get_timestamp().c_str());
    va_list args;
    va_start(args, format);
    vprintf(format, args);
    va_end(args);
    printf("\n");
}

void Logger::error(const char* format, ...)
{
    printf("%s ERR!: ", get_timestamp().c_str());
    va_list args;
    va_start(args, format);
    vprintf(format, args);
    va_end(args);
    printf("\n");
}
