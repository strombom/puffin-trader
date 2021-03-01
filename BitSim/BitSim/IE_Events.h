#pragma once
#include "pch.h"

#include "BitLib/DateTime.h"


class IE_Event
{
public:
    IE_Event(time_point_ms timestamp, float price, float price_max, float price_min, float delta, float delta_top, float delta_bot, std::chrono::milliseconds duration, float volume, int trade_count);

    time_point_ms timestamp;
    float price;
    float price_max;
    float price_min;

    float delta;
    float delta_top;
    float delta_bot;
    std::chrono::milliseconds duration;
    float volume;
    int trade_count;
};

using sptrIE_Event = std::shared_ptr<IE_Event>;


class IE_Events
{
public:
    void append(time_point_ms timestamp, float price, float price_max, float price_min, float delta, float delta_top, float delta_bot, std::chrono::milliseconds duration, float volume, int trade_count);

    std::vector<IE_Event> events;
};

using sptrIE_Events = std::shared_ptr<IE_Events>;

