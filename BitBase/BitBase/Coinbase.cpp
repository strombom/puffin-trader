#include "pch.h"

#include "BitLib/DateTime.h"
#include "BitLib/Logger.h"
#include "Coinbase.h"


Coinbase::Coinbase(sptrDatabase database) :
    database(database), state(CoinbaseState::idle), thread_running(true)
{
    Coinbase_tick = std::make_unique<CoinbaseTick>(database, std::bind(&Coinbase::tick_data_updated_callback, this));
    Coinbase_live = std::make_unique<CoinbaseLive>(database, std::bind(&Coinbase::tick_data_updated_callback, this));
    Coinbase_interval = std::make_unique<CoinbaseInterval>(database);

    main_loop_thread = std::make_unique<std::thread>(&Coinbase::main_loop, this);
    interval_update_thread = std::make_unique<std::thread>(&Coinbase::interval_update_worker, this);
}

void Coinbase::shutdown(void)
{
    logger.info("Coinbase::shutdown");
    {
        auto slock = std::scoped_lock{ state_mutex };
        state = CoinbaseState::shutdown;
    }
    logger.info("Coinbase::shutdown state = shutdown");
    Coinbase_tick->shutdown();
    Coinbase_live->shutdown();
    Coinbase_interval->shutdown();

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

void Coinbase::tick_data_updated_callback(void)
{
    interval_update_condition.notify_one();
}

void Coinbase::main_loop(void)
{
    while (state != CoinbaseState::shutdown) {
        {
            auto slock = std::scoped_lock{ state_mutex };

            if (state == CoinbaseState::idle) {
                auto tick_data_last_timestamp = database->get_attribute(BitBase::Coinbase::exchange_name, BitBase::Coinbase::symbols[0], "tick_data_last_timestamp", BitBase::Coinbase::first_timestamp);
                if (tick_data_last_timestamp < std::chrono::system_clock::now() - std::chrono::hours{ 1 }) {
                    // Last tick timestamp is more than 1 hour old, get data from Coinbase API
                    state = CoinbaseState::downloading_tick;
                    Coinbase_tick->start();
                }
                else {
                    state = CoinbaseState::downloading_live;
                    //Coinbase_live->start();
                }
            }
            else if (state == CoinbaseState::downloading_tick) {
                // Check if daily data is downloaded
                if (Coinbase_tick->get_state() == CoinbaseTickState::idle) {
                    state = CoinbaseState::idle;
                }
            }
            else if (state == CoinbaseState::downloading_live) {
                // Check if live data has stopped
                if (Coinbase_live->get_state() == CoinbaseLiveState::idle) {
                    state = CoinbaseState::idle;
                }
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

void Coinbase::interval_update_worker(void)
{
    while (state != CoinbaseState::shutdown) {
        {
            auto interval_update_lock = std::unique_lock<std::mutex>{ interval_update_mutex };
            interval_update_condition.wait(interval_update_lock);
        }

        Coinbase_interval->update();
    }
}
