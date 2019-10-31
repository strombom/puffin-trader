#pragma once

#include "boost/date_time.hpp"


class Logger
{
public:
    Logger(void);

    void info(const char* format, ...);
    void warn(const char* format, ...);
    void error(const char* format, ...);

private:

    std::string get_timestamp(void);

};

extern Logger logger;
