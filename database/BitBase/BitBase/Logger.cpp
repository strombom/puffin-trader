#include "Logger.h"
#include "DateTime.h"

#include <stdarg.h>

Logger logger;

Logger::Logger(void)
{

}

void Logger::info(const char* format, ...)
{
    std::scoped_lock lock(mutex);
    printf("%s INFO: ", DateTime::now().to_string().c_str());
    va_list args;
    va_start(args, format);
    vprintf(format, args);
    va_end(args);
    printf("\n");
}

void Logger::warn(const char* format, ...)
{
    std::scoped_lock lock(mutex);
    printf("%s WARN: ", DateTime::now().to_string().c_str());
    va_list args;
    va_start(args, format);
    vprintf(format, args);
    va_end(args);
    printf("\n");
}

void Logger::error(const char* format, ...)
{
    std::scoped_lock lock(mutex);
    printf("%s ERR!: ", DateTime::now().to_string().c_str());
    va_list args;
    va_start(args, format);
    vprintf(format, args);
    va_end(args);
    printf("\n");
}
