#pragma once
#include "pch.h"

#include "BitLib/Ticks.h"
#include "IE_Events.h"


class IE_Runner
{
public:
    IE_Runner(double delta, float initial_price, time_point_ms initial_timestamp);

    void step(sptrIE_Events& events, const sptrTicks& ticks, int tick_idx);
    void run(sptrIE_Events& events, const sptrTicks& ticks, time_point_ms timestamp_start, time_point_ms timestamp_end);


private:
    float delta;

    float current_price;

    time_point_ms ie_timestamp;
    int ie_trade_count;
    float ie_volume;
    float ie_start_price;
    float ie_max_price;
    float ie_min_price;
    float ie_delta_top;
    float ie_delta_bot;
};
