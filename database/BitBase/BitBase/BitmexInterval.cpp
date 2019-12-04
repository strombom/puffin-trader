
#include "Logger.h"
#include "BitmexInterval.h"
#include "BitmexConstants.h"

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

        const auto symbols = database->get_attribute(BitmexConstants::exchange_name, "symbols", std::unordered_set<std::string>{});
        for (auto&& symbol : symbols) {
            for (auto&& interval : intervals) {
                const auto interval_name = std::to_string(interval.count());
                const auto timeperiod = database->get_attribute(BitmexConstants::exchange_name, symbol + "_interval_" + interval_name + "_timestamp", BitmexConstants::bitmex_first_timestamp);
                const auto tick_idx = database->get_attribute(BitmexConstants::exchange_name, symbol + "_interval_" + interval_name + "_tick_idx", 0);

                auto count = 0;
                while (auto tick = database->get_tick(BitmexConstants::exchange_name, symbol, tick_idx + count)) {

                    ++count;
                    break;
                }

                const auto timeperiod_start = timeperiod;
                const auto timeperiod_end = timeperiod + interval;

                //auto ticks = database->get_tick_data(tick_idx, max_ticks_per_werk);
                //tick_count = (int) ticks->size();


                

                //auto ticks = database->tick_data_get

                //return;
            }
        }
    }
}
