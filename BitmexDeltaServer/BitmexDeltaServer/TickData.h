#pragma once

#include "DateTime.h"
#include "BitmexConstants.h"

#include <map>
#include <mutex>
#include <msgpack.hpp>


struct Tick
{
public:
    int timestamp_ms;
    float price;
    float volume;
    bool buy;

    time_point_ms timestamp(void)
    {
        return std::chrono::time_point<std::chrono::system_clock, std::chrono::milliseconds>{ std::chrono::milliseconds{ timestamp_ms } };
    }

    MSGPACK_DEFINE(timestamp_ms, price, volume, buy);
};


class TickData
{
public:
    TickData(void);

    static std::shared_ptr<TickData> create(void);

    std::unique_ptr<std::vector<Tick>> get(const std::string& symbol, time_point_ms timestamp);
    void append(const std::string& symbol, time_point_ms timestamp, float price, float volume, bool buy);

private:
    std::mutex tick_data_mutex;

    std::map<std::string, int> buffer_count;
    std::map<std::string, int> buffer_next_idx;
    std::map<std::string, std::unique_ptr<std::array<Tick, Bitmex::buffer_size>>> ticks;
};

using sptrTickData = std::shared_ptr<TickData>;
