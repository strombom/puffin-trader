#include "DelayedKlines.h"

#include <list>


DelayedKlines::DelayedKlines(const IntrinsicEvents& intrinsic_events, const TickData& tick_data)
{
    constexpr auto delay = 30ms;

    auto pending_klines = std::list<DelayedKline>{};

    auto first_ie_idx = size_t{ 0 };
    auto second_ie_idx = size_t{ 0 };

    // Find first and second idx
    while (intrinsic_events.events[first_ie_idx].timestamp <= intrinsic_events.events[0].timestamp) {
        first_ie_idx++;
    }
    while (intrinsic_events.events[second_ie_idx].timestamp <= intrinsic_events.events[first_ie_idx].timestamp) {
        second_ie_idx++;
    }

    while (second_ie_idx + 1 < intrinsic_events.events.size()) {
        pending_klines.push_back({
            intrinsic_events.events[first_ie_idx].timestamp + delay,
            intrinsic_events.events[second_ie_idx].timestamp + delay
        });

        const auto tmp_idx = second_ie_idx;
        while (second_ie_idx + 1 < intrinsic_events.events.size() && intrinsic_events.events[second_ie_idx].timestamp == intrinsic_events.events[tmp_idx].timestamp) {
            second_ie_idx++;
        }
        first_ie_idx = tmp_idx;
    }
    
    auto tick_idx = size_t{ 0 };
    for (auto &pending_kline : pending_klines) {
        pending_kline.open = tick_data.rows[tick_idx].price;
        pending_kline.low = std::numeric_limits<float>::max();

        while (tick_idx + 1 < tick_data.rows.size() && tick_data.rows[tick_idx].timestamp <= pending_kline.timestamp_close) {
            const auto& tick = tick_data.rows[tick_idx];
            pending_kline.high = std::max(pending_kline.high, tick.price);
            pending_kline.low = std::min(pending_kline.low, tick.price);
            pending_kline.close = tick.price;
            pending_kline.volume += tick.size;

            tick_idx++;
        }
    }
}
