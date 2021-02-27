#pragma once
#include "pch.h"

#include "BitLib/Ticks.h"
#include "IE_Events.h"


class IE_Runner
{
public:
    IE_Runner(double delta, float initial_price, time_point_ms initial_timestamp);

    void step(sptrIE_Events& events, const Tick &tick);

private:
    float delta;
    float previous_price;
    time_point_ms ie_timestamp;
    float ie_volume;
    float ie_max_price;
    float ie_min_price;
    float ie_delta_travel;
    int ie_trade_count;
    float current_ask;
    float current_bid;
};
