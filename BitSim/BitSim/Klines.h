#pragma once
#include "BitLib/DateTime.h"
#include "BitLib/BitBotConstants.h"

#include <filesystem>


struct Kline
{
    time_point_ms open_time;
    float open;
    float high;
    float low;
    float volume;
};

class Klines
{
public:
    Klines(void);

    time_point_ms get_timestamp_start(void) const;
    time_point_ms get_timestamp_end(void) const;

    void step_idx(time_point_ms timestamp);
    double get_open_price(const BitSim::Symbol& symbol) const;

private:
    bool load(const BitSim::Symbol& symbol);

    std::array<std::vector<Kline>, BitSim::symbols.size()> data;
    std::array<int, BitSim::symbols.size()> data_idx;
};
