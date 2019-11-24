#pragma once

#include <date.h>

using time_point = std::chrono::time_point<std::chrono::system_clock, std::chrono::seconds>;
using time_point_us = std::chrono::time_point<std::chrono::system_clock, std::chrono::microseconds>;

#define system_clock_now std::chrono::time_point_cast<std::chrono::milliseconds>(std::chrono::system_clock::now())
