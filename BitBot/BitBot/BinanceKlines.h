#pragma once
#include "pch.h"

#include "BitLib/DateTime.h"


class BinanceKline
{
public:
    BinanceKline(void) : open_time(), open(0), volume(0) {}
    BinanceKline(time_point_ms open_time, float open, float high, float low, float volume) :
        open_time(open_time), open(open), high(high), low(low), volume(volume) {}

    friend std::ostream& operator<<(std::ostream& stream, const BinanceKline& row);
    friend std::istream& operator>>(std::istream& stream, BinanceKline& row);

    time_point_ms open_time;
    float open;
    float high;
    float low;
    float volume;
};

class BinanceKlines
{
public:
    BinanceKlines(const std::string& symbol);
    BinanceKlines(const std::string& symbol, time_point_ms begin);
    
    // Copy constructor
    BinanceKlines(const BinanceKlines& binance_klines) : rows(binance_klines.rows) {}

    void load(void);
    void load(time_point_ms begin);
    void save(void) const;

    time_point_ms get_timestamp_begin(void) const;
    time_point_ms get_timestamp_end(void) const;

    friend std::ostream& operator<<(std::ostream& stream, const BinanceKlines& binance_klines_data);
    friend std::istream& operator>>(std::istream& stream, BinanceKlines& binance_klines_data);

    const std::string symbol;
    std::vector<BinanceKline> rows;
};

using sptrBinanceKlines = std::shared_ptr<BinanceKlines>;
