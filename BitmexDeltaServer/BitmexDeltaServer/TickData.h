#pragma once

#include "DateTime.h"

#include <map>


struct Tick
{
public:
    time_point_ms timestamp;
    float price;
    float volume;
    bool buy;
};


class TickData
{
public:

    void append(const std::string& symbol, time_point_ms timestamp, float price, float volume, bool buy);

private:
    std::map<std::string, std::vector<Tick>> ticks;

};

using sptrTickData = std::shared_ptr<TickData>;
