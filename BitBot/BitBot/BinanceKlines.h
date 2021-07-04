#pragma once
#include "pch.h"

#include "BitLib/DateTime.h"


class BinanceKline
{
public:
    BinanceKline(void) : timestamp(), open(0), volume(0) {}
    BinanceKline(time_point_ms timestamp, float open, float volume) : //, const step_prices_t& prices_buy, const step_prices_t& prices_sell) :
        timestamp(timestamp), open(open), volume(volume) {} //, prices_buy(prices_buy), prices_sell(prices_sell) {}

    friend std::ostream& operator<<(std::ostream& stream, const BinanceKline& row);
    friend std::istream& operator>>(std::istream& stream, BinanceKline& row);

    time_point_ms timestamp;
    float open;
    float volume;
};

class BinanceKlines
{
public:
    BinanceKlines(void) {}

    BinanceKlines(const std::string& file_path)
    {
        load(file_path);
    }

    // Copy constructor
    BinanceKlines(const BinanceKlines& binance_klines) : rows(binance_klines.rows) {}

    void load(const std::string& file_path);
    void save(const std::string& file_path) const;

    time_point_ms get_timestamp_begin(void) const;
    time_point_ms get_timestamp_end(void) const;

    friend std::ostream& operator<<(std::ostream& stream, const BinanceKlines& binance_klines_data);
    friend std::istream& operator>>(std::istream& stream, BinanceKlines& binance_klines_data);

    std::vector<BinanceKline> rows;
};

using sptrBinanceKlines = std::shared_ptr<BinanceKlines>;
