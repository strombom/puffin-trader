#include "pch.h"

#include "BitLib/DateTime.h"
#include "IE_Runner.h"


IE_Runner::IE_Runner(double delta_, float initial_price, time_point_ms initial_timestamp)
{
    delta = (float) delta_;
    current_price = initial_price;
    ie_start_price = initial_price;
    ie_max_price = initial_price;
    ie_min_price = initial_price;
    ie_timestamp = initial_timestamp;
    ie_volume = 0;
    ie_trade_count = 0;
    ie_delta_top = 0;
    ie_delta_bot = 0;
}

void IE_Runner::step(sptrIE_Events& events, const sptrTicks& ticks, int tick_idx)
{
    auto const static price_change_volume_threshold = 10.0f;

    const auto tick = &ticks->rows[tick_idx];

    auto static buy_accum_volume = 0.0f;
    if (tick->buy) {
        buy_accum_volume += tick->volume;
        if (buy_accum_volume > price_change_volume_threshold) {
            buy_accum_volume = 0.0f;
            current_price = std::min(current_price, tick->price);
        }
    }

    auto static sell_accum_volume = 0.0f;    
    if (!tick->buy)
    {
        sell_accum_volume += tick->volume;
        if (sell_accum_volume > price_change_volume_threshold) {
            sell_accum_volume = 0.0f;
            current_price = std::max(current_price, tick->price);
        }
    }

    auto delta_dir = float{};
    auto static previous_price = current_price;
    if (current_price > previous_price) {
        delta_dir = 1.0f;
    }
    else {
        delta_dir = -1.0f;
    }
    previous_price = current_price;

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

    ie_trade_count += 1;

    if (ie_delta_top + delta_down >= delta || ie_delta_bot + delta_up >= delta) {
        auto ie_duration = tick->timestamp - ie_timestamp;
        auto ie_price = float{};
        auto remaining_delta = float{};

        if (delta_dir == 1) {
            remaining_delta = ie_delta_bot + delta_up;
            ie_price = ie_min_price * (1.0f + (delta - ie_delta_bot));
        }
        else {
            remaining_delta = ie_delta_top + delta_down;
            ie_price = ie_max_price * (1.0f - (delta - ie_delta_top));
        }
        auto ie_delta = (ie_start_price - ie_price) / ie_start_price;
        if (ie_delta > delta) {
            auto a = 1;
        }
        while (remaining_delta >= 2 * delta) {
            if (delta_dir == 1) {
                ie_max_price = std::min(ie_max_price, ie_price);
            }
            else {
                ie_min_price = std::max(ie_min_price, ie_price);
            }

            events->append(tick->timestamp, ie_price, ie_max_price, ie_min_price, 0.0f, 0.0f, ie_delta, std::min(ie_delta_top, delta), std::min(ie_delta_bot, delta), ie_duration, ie_volume, ie_trade_count);

            const auto next_price = ie_price * (1.0f + delta_dir * delta);
            ie_start_price = ie_price;
            ie_volume = 0;
            ie_trade_count = 0;
            if (delta_dir == 1) {
                ie_max_price = next_price;
                ie_min_price = ie_price;
            }
            else {
                ie_max_price = ie_price;
                ie_min_price = next_price;
            }
            ie_delta_top = (ie_max_price - ie_start_price) / ie_start_price;
            ie_delta_bot = (ie_start_price - ie_min_price) / ie_start_price;
            ie_delta = (ie_start_price - next_price) / ie_start_price;
            ie_price = next_price;

            ie_duration = 0ms;

            remaining_delta -= delta;
        }

        ie_volume += tick->volume;
        events->append(tick->timestamp, ie_price, ie_max_price, ie_min_price, 0.0f, 0.0f, ie_delta, ie_delta_top, ie_delta_bot, ie_duration, ie_volume, ie_trade_count);

        ie_timestamp = tick->timestamp;
        ie_start_price = ie_price;
        ie_volume = 0;
        ie_trade_count = 0;
        ie_max_price = ie_price;
        ie_min_price = ie_price;
        ie_delta_top = 0;
        ie_delta_bot = 0;
    }
    else {
        ie_volume += tick->volume;
    }
}

void IE_Runner::run(sptrIE_Events& events, const sptrTicks& ticks, time_point_ms timestamp_start, time_point_ms timestamp_end)
{
    for (auto idx = 0; idx < ticks->rows.size(); idx++) {
    //for (const auto& tick : ticks->rows) {
        const auto tick = &ticks->rows[idx];
        if (tick->timestamp < timestamp_start) {
            continue;
        }
        if (tick->timestamp > timestamp_end) {
            break;
        }
        step(events, ticks, idx);
    }
}
