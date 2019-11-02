#pragma once

#include <string>
#include <boost/thread/mutex.hpp>

class Logger
{
public:
    Logger(void);

    void info(const char* format, ...);
    void warn(const char* format, ...);
    void error(const char* format, ...);

private:
    boost::mutex mutex;

    void print(const char* prefix, const char* format, ...);
};

extern Logger logger;
