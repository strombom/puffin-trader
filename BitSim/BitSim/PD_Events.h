#pragma once
#include "pch.h"

#include "BitLib/Ticks.h"
#include "BitLib/Intervals.h"


enum class PD_Direction {
    up,
    down
};

class PD_Event
{
public:

    PD_Event(time_point_ms timestamp, double price, PD_Direction direction) :
        timestamp(timestamp), price(price), direction(direction)
    {}

    time_point_ms timestamp;
    double price;
    PD_Direction direction;
};

class PD_Events
{
public:
    PD_Events(sptrTicks ticks);

    void make_events(sptrTicks ticks);
    void plot_events(sptrIntervals intervals);

private:
    std::chrono::milliseconds offset;

    std::vector<PD_Event> events;
    std::vector<PD_Event> events_offset;
    std::vector<PD_Event> tick_prices;
    std::vector<PD_Event> order_book_top;
    std::vector<PD_Event> order_book_bot;
};
