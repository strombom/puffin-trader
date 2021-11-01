#pragma once
#include "Symbols.h"
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
    void reset_idx(void);

    double get_open_price(const Symbol& symbol) const;
    double get_high_price(const Symbol& symbol) const;
    double get_low_price(const Symbol& symbol) const;
    double get_volume(const Symbol& symbol) const;

private:
    bool load(const Symbol& symbol);

    std::array<std::vector<Kline>, symbols.size()> data;
    std::array<int, symbols.size()> data_idx;
};
