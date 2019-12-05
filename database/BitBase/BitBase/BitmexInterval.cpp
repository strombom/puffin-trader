
#include "Logger.h"
#include "BitmexInterval.h"
#include "BitBaseConstants.h"

#include "date.h"


BitmexInterval::BitmexInterval(sptrDatabase database) :
    database(database),
    interval_data_thread_running(true)
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
            interval_data_condition.wait(interval_data_lock);
            if (!interval_data_thread_running) {
                break;
            }
        }

        const auto symbols = database->get_attribute(BitBase::Bitmex::exchange_name, "symbols", std::unordered_set<std::string>{});
        for (auto&& symbol : symbols) {
            for (auto&& interval : BitBase::Interval::intervals) {
                const auto interval_name = std::to_string(interval.count());
                const auto timeperiod = database->get_attribute(BitBase::Bitmex::exchange_name, symbol + "_interval_" + interval_name + "_timestamp", BitBase::Bitmex::first_timestamp);
                const auto tick_idx = database->get_attribute(BitBase::Bitmex::exchange_name, symbol + "_interval_" + interval_name + "_tick_idx", 0);

                const auto timeperiod_start = timeperiod;
                const auto timeperiod_end = timeperiod + interval;

                const auto last_price = database->get_tick(BitBase::Bitmex::exchange_name, symbol, std::max(0, tick_idx - 1))->price;

                auto buys = std::vector<std::pair<float, float>>{};
                auto sells = std::vector<std::pair<float, float>>{};

                auto tick_count = 0;
                while (auto tick = database->get_tick(BitBase::Bitmex::exchange_name, symbol, tick_idx + tick_count)) {
                    if (tick->timestamp >= timeperiod_end) {
                        break;
                    }
                    if (tick->buy) {
                        buys.push_back({ tick->price, tick->volume });
                    }
                    else {
                        sells.push_back({ tick->price, tick->volume });
                    }
                    ++tick_count;
                }

                // Sort by volume
                std::sort(buys.begin(), buys.end());
                std::sort(sells.begin(), sells.end(), std::greater<std::pair<float, float>>());

                auto prices_buy = std::array<float, 6>{ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
                auto prices_sell = std::array<float, 6>{ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
                auto accum_vol_buy = 0.0f;
                auto accum_vol_sell = 0.0f;

                auto step_idx = 0;
                for (auto&& buy : buys) {
                    accum_vol_buy += buy.second;
                    while (step_idx < BitBase::Interval::steps.size() && accum_vol_buy > BitBase::Interval::steps[step_idx]) {
                        prices_buy[step_idx] = buy.first;
                        ++step_idx;
                    }
                }

                step_idx = 0;
                for (auto&& sell : sells) {
                    accum_vol_sell += sell.second;
                    while (step_idx < BitBase::Interval::steps.size() && accum_vol_sell > BitBase::Interval::steps[step_idx]) {
                        prices_sell[step_idx] = sell.first;
                        ++step_idx;
                    }
                }

                database->
                //timeperiod_start, last_price, accum_vol_buy, accum_vol_sell, prices_buy, prices_sell

                if (buys.size() > 0 || sells.size() > 0) {
                    logger.info("ok");
                }


                if (tick_count > 0) {
                    logger.info("reading %s (%s) (%d)", symbol.c_str(), date::format("%F %T", timeperiod_start).c_str(), tick_count);
                    logger.info("reading ok (%d)", tick_count);
                }
                
                logger.info("reading end (%d)", tick_count);
                break;
            }
        }
    }
}
