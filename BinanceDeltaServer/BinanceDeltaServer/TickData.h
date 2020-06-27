#pragma once

#include "DateTime.h"
#include "BinanceConstants.h"

#include <map>
#include <mutex>
#include <msgpack.hpp>


struct Tick
{
public:
    unsigned long long timestamp_ms;
    float price;
    float volume;
    bool buy;
    long long trade_id;

    time_point_ms timestamp(void)
    {
        return std::chrono::time_point<std::chrono::system_clock, std::chrono::milliseconds>{ std::chrono::milliseconds{ timestamp_ms } };
    }

    MSGPACK_DEFINE(timestamp_ms, price, volume, buy, trade_id);
};


class TickData
{
public:
    TickData(void);

    static std::shared_ptr<TickData> create(void);

    std::unique_ptr<std::vector<Tick>> get(const std::string& symbol, time_point_ms timestamp, int max_rows);
    void append(const std::string& symbol, time_point_ms timestamp, float price, float volume, bool buy, long long trade_id);

private:
    std::mutex tick_data_mutex;

    std::map<std::string, int> buffer_count;
    std::map<std::string, int> buffer_next_idx;
    std::map<std::string, std::unique_ptr<std::array<Tick, Binance::buffer_size>>> ticks;
};

using sptrTickData = std::shared_ptr<TickData>;
