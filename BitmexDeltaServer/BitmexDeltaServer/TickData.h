#pragma once

#include "DateTime.h"
#include "BitmexConstants.h"

#include <map>
#include <mutex>


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
