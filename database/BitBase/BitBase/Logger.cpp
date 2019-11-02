#include "Logger.h"
#include "DateTime.h"

#include <stdarg.h>

Logger logger;

Logger::Logger(void)
{

}

void Logger::print(const char* prefix, const char* format, ...)
{
    boost::mutex::scoped_lock lock(mutex);
    printf("%s %s: ", DateTime().to_string().c_str(), prefix);
    va_list args;
    va_start(args, format);
    vprintf(format, args);
    va_end(args);
    printf("\n");
}

void Logger::info(const char* format, ...)
{
    va_list args;
    va_start(args, format);
    print("INFO", format, args);
    va_end(args);
}

void Logger::warn(const char* format, ...)
{
    va_list args;
    va_start(args, format);
    print("WARN", format, args);
    va_end(args);
}

void Logger::error(const char* format, ...)
{
    va_list args;
    va_start(args, format);
    print("ERR!", format, args);
    va_end(args);
}
