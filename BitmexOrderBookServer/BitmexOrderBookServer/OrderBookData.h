#pragma once

#include "DateTime.h"
#include "BitmexConstants.h"

#include <map>
#include <mutex>
#include <msgpack.hpp>


struct OrderBook
{
public:
    unsigned long long timestamp_ms;
    float bid_price;
    float bid_volume;
    float ask_price;
    float ask_volume;

    time_point_ms timestamp(void)
    {
        return std::chrono::time_point<std::chrono::system_clock, std::chrono::milliseconds>{ std::chrono::milliseconds{ timestamp_ms } };
    }

    MSGPACK_DEFINE(timestamp_ms, bid_price, bid_volume, ask_price, ask_volume);
};


class OrderBookData
{
public:
    OrderBookData(void);

    static std::shared_ptr<OrderBookData> create(void);

    std::unique_ptr<std::vector<OrderBook>> get(const std::string& symbol, time_point_ms timestamp, int max_rows);
    void append(const std::string& symbol, time_point_ms timestamp, float bid_price, float bid_volume, float ask_price, float ask_volume);

private:
    std::mutex tick_data_mutex;

    std::map<std::string, int> buffer_count;
    std::map<std::string, int> buffer_next_idx;
    std::map<std::string, std::unique_ptr<std::array<OrderBook, Bitmex::buffer_size>>> order_book_ticks;
};

using sptrOrderBookData = std::shared_ptr<OrderBookData>;
