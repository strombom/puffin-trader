#include "DelayedKlines.h"

#include <list>


DelayedKlines::DelayedKlines(const IntrinsicEvents& intrinsic_events, const TickData& tick_data)
{
    constexpr auto delay = 30ms;

    auto pending_klines = std::list<DelayedKline>{};

    auto start_tick_idx = size_t{ 0 };

    auto previous_timestamp = tick_data.rows[0].timestamp;

    for (auto ie_idx = 0; ie_idx < intrinsic_events.events.size() - 1; ie_idx++) {

        const auto& event = intrinsic_events.events[ie_idx];
        
        // timestamp_open, timestamp_close, open, close, high, low, volume
        pending_klines.push_back({
            previous_timestamp + delay,
            event.timestamp + delay
        });

        std::cout << "Kline  " << DateTime::to_string(previous_timestamp) << " - " << DateTime::to_string(event.timestamp) << "  " << event.price << std::endl;

        previous_timestamp = event.timestamp;


        /*
        
        auto tick_idx = start_tick_idx;
        const auto start_timestamp = tick_data.rows[start_tick_idx].timestamp + delay;
        while (tick_idx < tick_data.rows.size() && tick_data.rows[tick_idx].timestamp < start_timestamp + delay) {
            tick_idx++;
        }

        const auto open = tick_data.rows[tick_idx].price;
        auto low = open;
        auto high = open;
        auto volume = tick_data.rows[tick_idx].size;


        const auto end_timestamp = intrinsic_events.events[ie_idx].timestamp + delay;
        while (tick_idx + 1 < tick_data.rows.size() && tick_data.rows[tick_idx + 1].timestamp < end_timestamp) {
            tick_idx++;
            const auto price = tick_data.rows[tick_idx].price;
            low = std::min(low, price);
            high = std::max(high, price);
            volume += tick_data.rows[tick_idx].size;
        }

        const auto close = tick_data.rows[tick_idx].price;

        //timestamp, open, close, high, low, volume
        klines.push_back({ end_timestamp, open, close, high, low, volume }) ;
        std::cout << "Kline  " << DateTime::to_string(start_timestamp) << " - " << DateTime::to_string(end_timestamp);
        std::cout << "  " << open << ", " << close << ", " << high << ", " << low << ", " << volume << std::endl << std::endl;

        if (volume != 0) {
            start_tick_idx = tick_idx;
        }
        */

        //while (tick_idx < tick_data.rows.size() && tick_data.rows[tick_idx].timestamp < 1) {

        //auto tick_idx = intrinsic_events.events[ie_idx].tick_id;
        //while (tick_idx < tick_data.rows.size() && tick_data.rows[tick_idx].timestamp > 1) {
        //    tick_idx++;
        //}

    }

    /*
    auto ie_idx = 0;
    for (const auto& tick : tick_data.rows) {
        while (intrinsic_events.events[ie_idx].timestamp < std::get<0>(tick)) {
            ie_idx++;
        }
        auto a = 1;
    }
    */
}
