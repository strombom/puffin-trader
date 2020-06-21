#include "pch.h"

#include "BitLib/DateTime.h"
#include "BitLib/Logger.h"
#include "Binance.h"


Binance::Binance(sptrDatabase database) :
    database(database), state(BinanceState::idle), thread_running(true)
{
    binance_tick = std::make_unique<BinanceTick>(database, std::bind(&Binance::tick_data_updated_callback, this));
    binance_live = std::make_unique<BinanceLive>(database, std::bind(&Binance::tick_data_updated_callback, this));
    binance_interval = std::make_unique<BinanceInterval>(database);

    main_loop_thread = std::make_unique<std::thread>(&Binance::main_loop, this);
    interval_update_thread = std::make_unique<std::thread>(&Binance::interval_update_worker, this);
}

void Binance::shutdown(void)
{
    logger.info("Binance::shutdown");
    {
        auto slock = std::scoped_lock{ state_mutex };
        state = BinanceState::shutdown;
    }
    logger.info("Binance::shutdown state = shutdown");
    binance_tick->shutdown();
    binance_live->shutdown();
    binance_interval->shutdown();

    try {
        main_loop_thread->join();
    }
    catch (...) {}

    try {
        interval_update_condition.notify_all();
        interval_update_thread->join();
    }
    catch (...) {}
}

void Binance::tick_data_updated_callback(void)
{
    interval_update_condition.notify_one();
}

void Binance::main_loop(void)
{
    while (state != BinanceState::shutdown) {
        {
            auto slock = std::scoped_lock{ state_mutex };

            if (state == BinanceState::idle) {
                auto tick_data_last_timestamp = database->get_attribute("BITMEX", "XBTUSD", "tick_data_last_timestamp", BitBase::Binance::first_timestamp);
                if (tick_data_last_timestamp < std::chrono::system_clock::now() - std::chrono::hours{ 1 }) {
                    // Last tick timestamp is more than 1 hour old, get data from Binance API
                    state = BinanceState::downloading_tick;
                    binance_tick->start();
                }
                else {
                    state = BinanceState::downloading_live;
                    binance_live->start();
                }

            }
            else if (state == BinanceState::downloading_tick) {
                // Check if daily data is downloaded
                if (binance_tick->get_state() == BinanceTickState::idle) {
                    state = BinanceState::idle;
                }
            }
            else if (state == BinanceState::downloading_live) {
                // Check if live data has stopped
                if (binance_live->get_state() == BinanceLiveState::idle) {
                    state = BinanceState::idle;
                }
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
}

void Binance::interval_update_worker(void)
{
    while (state != BinanceState::shutdown) {
        {
            auto interval_update_lock = std::unique_lock<std::mutex>{ interval_update_mutex };
            interval_update_condition.wait(interval_update_lock);
        }

        binance_interval->update();
    }
}
