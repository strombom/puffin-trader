#pragma once
#include "pch.h"

#include <string>
#include <mutex>


class Logger
{
public:
    Logger(void);

    void info(const char* format, ...);
    void warn(const char* format, ...);
    void error(const char* format, ...);

private:
    std::mutex mutex;
};

extern Logger logger;
