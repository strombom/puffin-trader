#include "pch.h"
#include "IE_Runner.h"


IE_Runner::IE_Runner(double delta_, float initial_price, time_point_ms initial_timestamp)
{
    delta = (float) delta_;
    current_ask = initial_price;
    current_bid = initial_price;
    previous_price = initial_price;
    ie_max_price = initial_price;
    ie_min_price = initial_price;
    ie_delta_travel = 0;
    ie_volume = 0;
    ie_timestamp = initial_timestamp;
    ie_trade_count = 0;
}

void IE_Runner::step(sptrIE_Events& events, const Tick& tick)
{
    ie_trade_count += 1;

    if (tick.buy) {
        current_ask = tick.price;
    }
    else {
        current_bid = tick.price;
    }

    ie_max_price = std::max(ie_max_price, tick.price);
    ie_min_price = std::min(ie_min_price, tick.price);

    const auto current_price = std::max(std::min(previous_price, current_ask), current_bid);
    auto delta_travel = std::abs(current_price - previous_price) / previous_price;

    if (ie_delta_travel + delta_travel < delta) {
        // No intrinsic event
        ie_delta_travel += delta_travel;
        ie_volume += tick.volume;
    }
    else if (ie_delta_travel + delta_travel >= delta) {
        // One intrinsic event
        const auto delta_travel_remaining = delta - ie_delta_travel;
        const auto delta_travel_extra = delta_travel - delta_travel_remaining;
        const auto delta_price = current_price * (1.0f + delta_travel_remaining);

        const auto ie_duration = tick.timestamp - ie_timestamp;
        const auto ie_spread = ie_max_price - ie_min_price;

        ie_volume += delta_travel_remaining / delta_travel * tick.volume;

        events->append(ie_duration, ie_volume, ie_spread, ie_trade_count);
        
        ie_delta_travel = delta_travel - delta_travel_remaining;
        ie_volume = delta_travel_extra / delta_travel * tick.volume;
        ie_timestamp = tick.timestamp;
        ie_trade_count = 0;

        std::cout << "a " << 1 << std::endl;
    }
    else {
        // Multiple intrinsic events


        std::cout << "b " << 2 << std::endl;
    }

    //while (dc_delta_travel >= delta) {
    //    delta_travel = 
    //    events.append();
    //    dc_delta_travel -= delta;
    //}

    previous_price = current_price;
}
