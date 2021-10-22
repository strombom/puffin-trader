#pragma once
#include "BitLib/DateTime.h"
#include "BitLib/BitBotConstants.h"

#include <filesystem>


struct Kline
{
    time_point_ms open_time;
    float open_price;
    float volume;
};

class Klines
{
public:
    Klines(void);

    time_point_ms get_timestamp_start(void) const;
    time_point_ms get_timestamp_end(void) const;

    void step_idx(time_point_ms timestamp);

    std::array<std::vector<Kline>, BitBot::symbols.size()> data;

private:
    void save(const BitBot::Symbol symbol);
    bool load(const BitBot::Symbol symbol);

    std::array<int, BitBot::symbols.size()> data_idx;
};
