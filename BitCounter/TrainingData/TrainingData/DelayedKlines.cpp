#include "DelayedKlines.h"

#include <list>


DelayedKlines::DelayedKlines(const IntrinsicEvents& intrinsic_events, const TickData& tick_data)
{
    constexpr auto delay = 30ms;

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
        klines.push_back({
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
    for (auto &pending_kline : klines) {
        pending_kline.open = tick_data.rows[tick_idx].price;
        pending_kline.high = pending_kline.open;
        pending_kline.low = pending_kline.open;
        pending_kline.close = pending_kline.open;

        while (tick_idx + 1 < tick_data.rows.size() && tick_data.rows[tick_idx].timestamp <= pending_kline.timestamp_close) {
            const auto& tick = tick_data.rows[tick_idx];
            pending_kline.high = std::max(pending_kline.high, tick.price);
            pending_kline.low = std::min(pending_kline.low, tick.price);
            pending_kline.close = tick.price;
            pending_kline.volume += tick.size;
            tick_idx++;
        }
}

void DelayedKlines::save_csv(std::string file_path)
{
    auto csv_file = std::ofstream{ file_path, std::ios::binary };
    csv_file << "\"timestamp_open\",\"timestamp_close\",\"open\",\"close\",\"high\",\"low\",\"volume\"\n";
    csv_file << std::fixed;
    for (const auto& kline : klines) {
        const auto timestamp_open = kline.timestamp_open.time_since_epoch().count() / 1000000.0;
        const auto timestamp_close = kline.timestamp_close.time_since_epoch().count() / 1000000.0;
        csv_file.precision(6);
        csv_file << timestamp_open << ",";
        csv_file << timestamp_close << ",";
        csv_file.precision(2);
        csv_file << kline.open << ",";
        csv_file << kline.close << ",";
        csv_file << kline.high << ",";
        csv_file << kline.low << ",";
        csv_file.precision(3);
        csv_file << kline.volume << "\n";
    }
    csv_file.close();
}
