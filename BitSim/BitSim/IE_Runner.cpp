#include "pch.h"

#include "BitLib/DateTime.h"
#include "IE_Runner.h"


IE_Runner::IE_Runner(double delta_, float initial_price, time_point_ms initial_timestamp)
{
    delta = (float) delta_;
    //current_ask = initial_price;
    //current_bid = initial_price;
    current_price = initial_price;
    //previous_price = initial_price;
    ie_start_price = initial_price;
    ie_max_price = initial_price;
    ie_min_price = initial_price;
    ie_volume = 0;
    ie_timestamp = initial_timestamp;
    ie_trade_count = 0;

    ie_delta_top = 0;
    ie_delta_bot = 0;
    //ie_delta_travel = 0;
}

template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

void IE_Runner::step(sptrIE_Events& events, const Tick& tick)
{
    ie_trade_count += 1;

    if (tick.buy) {
        current_price = std::min(current_price, tick.price);
    }
    else {
        current_price = std::max(current_price, tick.price);
    }

    if (current_price > ie_max_price) {
        ie_max_price = current_price;
        ie_delta_top = (ie_max_price - ie_start_price) / ie_start_price;
    }
    else if (current_price < ie_min_price) {
        ie_min_price = current_price;
        ie_delta_bot = (ie_start_price - ie_min_price) / ie_start_price;
    }

    auto delta_down = (ie_max_price - current_price) / ie_max_price; // Delta from top
    auto delta_up = (current_price - ie_min_price) / ie_min_price;   // Delta from bottom

    auto delta_dir = float{};
    if (ie_delta_top + delta_down > ie_delta_bot + delta_up) {
        // down from max-price
        delta_dir = 1.0f;
    }
    else {
        // up from min-price
        delta_dir = -1.0;
    }

    if (ie_delta_top + delta_down >= delta || ie_delta_bot + delta_up >= delta) {
        auto ie_duration = tick.timestamp - ie_timestamp;
        auto ie_price = float{};
        auto remaining_delta = float{};

        if (delta_dir == 1) {
            remaining_delta = ie_delta_top + delta_down;
            ie_price = ie_max_price * (1.0f - delta_down);
        }
        else {
            remaining_delta = ie_delta_bot + delta_up;
            ie_price = ie_min_price * (1.0f + delta_up);
        }
        auto ie_delta = (ie_start_price - ie_price) / ie_start_price;
        if (ie_delta > delta) {
            auto a = 1;
        }
        while (remaining_delta >= 2 * delta) {
            if (ie_delta > delta || ie_delta_top > delta || ie_delta_bot > delta) {
                auto a = 1;
            }
            events->append(tick.timestamp, ie_price, ie_max_price, ie_min_price, ie_delta, std::min(ie_delta_top, delta), std::min(ie_delta_bot, delta), ie_duration, ie_volume, ie_trade_count);

            const auto next_price = ie_price * (1.0f - delta_dir * delta);
            ie_start_price = ie_price;
            ie_volume = 0;
            ie_trade_count = 0;
            if (delta_dir == 1) {
                ie_max_price = ie_price;
                ie_min_price = next_price;
            }
            else {
                ie_max_price = next_price;
                ie_min_price = ie_price;
            }
            ie_delta_top = (ie_max_price - ie_start_price) / ie_start_price;
            ie_delta_bot = (ie_start_price - ie_min_price) / ie_start_price;
            ie_delta = (ie_start_price - next_price) / ie_start_price;
            ie_price = next_price;

            ie_duration = 0ms;

            remaining_delta -= delta;
        }

        ie_volume += tick.volume;
        events->append(tick.timestamp, ie_price, ie_max_price, ie_min_price, ie_delta, ie_delta_top, ie_delta_bot, ie_duration, ie_volume, ie_trade_count);

        ie_timestamp = tick.timestamp;
        ie_start_price = ie_price;
        ie_volume = 0;
        ie_trade_count = 0;
        ie_max_price = ie_price;
        ie_min_price = ie_price;
        ie_delta_top = 0;
        ie_delta_bot = 0;
    }
    else {
        ie_volume += tick.volume;
    }

    /*
    else if (ie_delta_bot + delta_up >= delta) {
        auto remaining_delta = ie_delta_bot + delta_up;
        auto ie_price = ie_min_price * (1.0f + std::min(remaining_delta, delta_up));
        auto ie_duration = tick.timestamp - ie_timestamp;

        while (remaining_delta >= 2 * delta) {
            auto ie_delta = (ie_start_price - ie_price) / ie_start_price;
            events->append(tick.timestamp, ie_price, ie_delta, ie_delta_top, ie_delta_bot, ie_duration, ie_volume, ie_trade_count);

            ie_timestamp = tick.timestamp;
            ie_start_price = ie_price;
            ie_volume = 0;
            ie_trade_count = 0;
            ie_max_price = ie_price;
            ie_min_price = ie_price;
            ie_delta_top = delta;
            ie_delta_bot = 0;
            ie_price = ie_price * (1.0f + delta);

            ie_duration = 0ms;

            remaining_delta -= delta;
        }

        ie_volume += tick.volume;

        auto ie_delta = (ie_start_price - ie_price) / ie_start_price;
        events->append(tick.timestamp, ie_price, ie_delta, ie_delta_top, ie_delta_bot, ie_duration, ie_volume, ie_trade_count);

        ie_timestamp = tick.timestamp;
        ie_start_price = ie_price;
        ie_volume = 0;
        ie_trade_count = 0;
        ie_max_price = ie_price;
        ie_min_price = ie_price;
        ie_delta_top = 0;
        ie_delta_bot = 0;
    }
    */

    return;


    /*
    auto delta_travel = std::abs(current_price - previous_price) / previous_price;

    if (ie_delta_travel + delta_travel < delta) {
        // No intrinsic event
        ie_delta_travel += delta_travel;
        ie_volume += tick.volume;
    }
    else if (ie_delta_travel + delta_travel >= delta) {
        // One intrinsic event
        const auto direction = signbit(current_price - previous_price) ? -1.0f : 1.0f;
        const auto delta_travel_remaining = delta - ie_delta_travel;
        const auto delta_travel_extra = delta_travel - delta_travel_remaining;
        const auto delta_price = current_price * (1.0f + direction * delta_travel_remaining);
        const auto ie_delta = (delta_price - ie_start_price) / delta_price;
        const auto ie_duration = tick.timestamp - ie_timestamp;
        const auto ie_spread = (ie_max_price - ie_min_price) / tick.price;

        ie_volume += delta_travel_remaining / delta_travel * tick.volume;

        if (false) {
            std::cout << "timestamp(" << DateTime::to_string_iso_8601(tick.timestamp) << ")";
            std::cout << " price(" << tick.price << ")";
            std::cout << " delta(" << ie_delta * 100000 << ")";
            std::cout << " time(" << ie_duration.count() << ")";
            std::cout << " vol(" << ie_volume << ")";
            std::cout << " spread(" << ie_spread << ")";
            std::cout << " trade count(" << ie_trade_count << ")";
            std::cout << std::endl;
        }
        events->append(tick.timestamp, tick.price, ie_delta, ie_duration, ie_volume, ie_spread, ie_trade_count);
        
        // Next intrinsic event
        ie_delta_travel = delta_travel - delta_travel_remaining;
        ie_volume = delta_travel_extra / delta_travel * tick.volume;
        ie_timestamp = tick.timestamp;
        ie_trade_count = 0;
        ie_start_price = tick.price;
        ie_max_price = tick.price;
        ie_min_price = tick.price;
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
    */
}
