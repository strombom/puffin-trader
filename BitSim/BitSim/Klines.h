#pragma once
#include "BitLib/DateTime.h"

#include <filesystem>
#include <unordered_map>


struct Kline
{
    time_point_ms open_time;
    float open_price;
    float volume;
};

class Klines
{
public:
    Klines();

    time_point_ms get_timestamp_start(void) const;
    time_point_ms get_timestamp_end(void) const;

    std::unordered_map<std::string, std::vector<Kline>> data;

private:
    void save(const std::string symbol);
    bool load(const std::string symbol);
};
