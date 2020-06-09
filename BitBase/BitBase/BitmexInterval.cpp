#include "pch.h"

#include "Logger.h"
#include "BitmexInterval.h"
#include "Intervals.h"

#include <algorithm>


BitmexInterval::BitmexInterval(sptrDatabase database) :
    database(database), interval_data_thread_running(true)
{
    interval_data_worker_thread = std::make_unique<std::thread>(&BitmexInterval::interval_data_worker, this);
}

void BitmexInterval::shutdown(void)
{
    interval_data_thread_running = false;
    interval_data_condition.notify_all();

    try {
        interval_data_worker_thread->join();
    }
    catch (...) {}
}

void BitmexInterval::update(void)
{
    interval_data_condition.notify_one();
}

void BitmexInterval::interval_data_worker(void)
{
    while (interval_data_thread_running) {
        {
            auto interval_data_lock = std::unique_lock<std::mutex>{ interval_data_mutex };
            interval_data_condition.wait_for(interval_data_lock, std::chrono::milliseconds{ 500 });
            if (!interval_data_thread_running) {
                break; 
            }
        }

        const auto symbols = database->get_attribute(BitBase::Bitmex::exchange_name, "symbols", std::unordered_set<std::string>{});
        for (auto&& symbol : symbols) {
            if (std::find(BitBase::Bitmex::symbols.begin(), BitBase::Bitmex::symbols.end(), symbol) == BitBase::Bitmex::symbols.end()) {
                // Symbol is not enabled, continue with next symbol
                continue;
            }
            for (auto&& interval : BitBase::Bitmex::Interval::intervals) {
                make_interval(symbol, interval);
            }
        }
    }
}

void BitmexInterval::make_interval(const std::string& symbol, std::chrono::milliseconds interval)
{
    const auto interval_name = std::to_string(interval.count());
    auto timestamp = database->get_attribute(BitBase::Bitmex::exchange_name, symbol + "_interval_" + interval_name + "_next_timestamp", BitBase::Bitmex::first_timestamp);
    const auto last_timestamp = timestamp + interval * (BitBase::Bitmex::Interval::batch_size - 1);
    auto next_tick_idx = database->get_attribute(BitBase::Bitmex::exchange_name, symbol + "_interval_" + interval_name + "_next_tick_idx", 0);
    auto tick_table = database->open_tick_table_read(BitBase::Bitmex::exchange_name, symbol);
    const auto last_tick = tick_table->get_tick(std::max(0, next_tick_idx - 1));
    if (!last_tick) {
        // End of file
        return;
    }

    auto last_price = last_tick->price;
    auto tick = tick_table->get_tick(next_tick_idx);

    auto intervals_data = Intervals{ timestamp, interval };
    const auto timer = Timer{};

    while (timestamp <= last_timestamp && timer.elapsed() < BitBase::Bitmex::Interval::batch_timeout) {
        auto buys = std::vector<std::pair<float, float>>{};
        auto sells = std::vector<std::pair<float, float>>{};

        auto valid_interval = false;
        while (tick) {
            if (tick->timestamp >= timestamp + interval) {
                valid_interval = true;
                break;
            }
            if (tick->buy) {
                buys.push_back({ tick->price, tick->volume });
            }
            else {
                sells.push_back({ tick->price, tick->volume });
            }
            last_price = tick->price;
            tick = tick_table->get_next_tick();
            ++next_tick_idx;
        }

        if (!valid_interval) {
            // End of tick data, do not save current (incomplete) interval
            break;
        }

        // Sort buys and sells by volume
        std::sort(buys.begin(), buys.end(), std::less<std::pair<float, float>>());
        std::sort(sells.begin(), sells.end(), std::greater<std::pair<float, float>>());

        auto prices_buy = step_prices_t{ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
        auto prices_sell = step_prices_t{ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
        auto accum_vol_buy = 0.0f;
        auto accum_vol_sell = 0.0f;

        auto step_idx = 0;
        for (auto&& buy : buys) {
            accum_vol_buy += buy.second;
            while (step_idx < BitBase::Bitmex::Interval::steps.size() && accum_vol_buy > BitBase::Bitmex::Interval::steps[step_idx]) {
                prices_buy[step_idx] = buy.first;
                ++step_idx;
            }
        }

        step_idx = 0;
        for (auto&& sell : sells) {
            accum_vol_sell += sell.second;
            while (step_idx < BitBase::Bitmex::Interval::steps.size() && accum_vol_sell > BitBase::Bitmex::Interval::steps[step_idx]) {
                prices_sell[step_idx] = sell.first;
                ++step_idx;
            }
        }

        intervals_data.rows.push_back({ last_price, accum_vol_buy, accum_vol_sell, prices_buy, prices_sell });
        timestamp += interval;
    } 

    database->extend_interval_data(BitBase::Bitmex::exchange_name, symbol, interval, intervals_data, timestamp, next_tick_idx);
}
