#pragma once
#include "pch.h"

#include "BitLib/AggTicks.h"


enum class PD_Direction {
    up,
    down
};

class PD_OrderBookBuffer {
public:
    PD_OrderBookBuffer(void);

    void update(time_point_ms timestamp, double price);

    std::tuple<double, double> get_price(time_point_ms timestamp);

private:
    static constexpr int size = 1000;
    std::array<time_point_ms, size> timestamps;
    std::array<double, size> prices;
    int length;
    int next_idx;

    double order_book_bottom;
};

class PD_OrderBook {
public:
    PD_OrderBook(time_point_ms timestamp, double price);

    bool update(time_point_ms timestamp, double price, PD_Direction direction);

    PD_OrderBookBuffer buffer;
private:
};

class PD_Event
{
public:
    PD_Event(time_point_ms timestamp, double price, PD_Direction direction, size_t agg_tick_idx) :
        timestamp(timestamp), price(price), direction(direction), agg_tick_idx(agg_tick_idx)
    {}

    time_point_ms timestamp;
    double price;
    PD_Direction direction;
    size_t agg_tick_idx;
};

using sptrPD_Event = std::shared_ptr<PD_Event>;


class PD_Events
{
public:
    PD_Events(sptrAggTicks agg_ticks);

    std::vector<PD_Event> events;

    /*
    PD_Events(const Tick& first_tick);
    PD_Events(time_point_ms timestamp, const Interval& first_interval);

    PD_Events(sptrTicks ticks);
    PD_Events(sptrIntervals intervals);

    sptrPD_Event step(const Tick& tick);
    sptrPD_Event step(const Interval& intervals);

    void plot_events(sptrIntervals intervals);
    */

private:
    //PD_Direction last_direction;
    //PD_OrderBook order_book;


    size_t event_idx;

    //std::vector<PD_Event> events_offset;
    //std::vector<PD_Event> tick_prices;
    //std::vector<PD_Event> order_book_top;
    //std::vector<PD_Event> order_book_bot;
};

using sptrPD_Events = std::shared_ptr<PD_Events>;
