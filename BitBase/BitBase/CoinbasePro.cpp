#include "pch.h"

#include "BitLib/DateTime.h"
#include "BitLib/Logger.h"
#include "CoinbasePro.h"


CoinbasePro::CoinbasePro(sptrDatabase database) :
    database(database), state(CoinbaseProState::idle), thread_running(true)
{
    CoinbasePro_tick = std::make_unique<CoinbaseProTick>(database, std::bind(&CoinbasePro::tick_data_updated_callback, this));
    CoinbasePro_live = std::make_unique<CoinbaseProLive>(database, std::bind(&CoinbasePro::tick_data_updated_callback, this));
    CoinbasePro_interval = std::make_unique<CoinbaseProInterval>(database);

    main_loop_thread = std::make_unique<std::thread>(&CoinbasePro::main_loop, this);
    interval_update_thread = std::make_unique<std::thread>(&CoinbasePro::interval_update_worker, this);
}

void CoinbasePro::shutdown(void)
{
    logger.info("CoinbasePro::shutdown");
    {
        auto slock = std::scoped_lock{ state_mutex };
        state = CoinbaseProState::shutdown;
    }
    logger.info("CoinbasePro::shutdown state = shutdown");
    CoinbasePro_tick->shutdown();
    CoinbasePro_live->shutdown();
    CoinbasePro_interval->shutdown();

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

void CoinbasePro::tick_data_updated_callback(void)
{
    interval_update_condition.notify_one();
}

void CoinbasePro::main_loop(void)
{
    while (state != CoinbaseProState::shutdown) {
        {
            auto slock = std::scoped_lock{ state_mutex };

            if (state == CoinbaseProState::idle) {
                auto tick_data_last_timestamp = database->get_attribute(BitBase::CoinbasePro::exchange_name, BitBase::CoinbasePro::symbols[0], "tick_data_last_timestamp", BitBase::CoinbasePro::first_timestamp);
                if (tick_data_last_timestamp < std::chrono::system_clock::now() - std::chrono::hours{ 1 }) {
                    // Last tick timestamp is more than 1 hour old, get data from CoinbasePro API
                    state = CoinbaseProState::downloading_tick;
                    CoinbasePro_tick->start();
                }
                else {
                    state = CoinbaseProState::downloading_live;
                    //CoinbasePro_live->start();
                }
            }
            else if (state == CoinbaseProState::downloading_tick) {
                // Check if daily data is downloaded
                if (CoinbasePro_tick->get_state() == CoinbaseProTickState::idle) {
                    state = CoinbaseProState::idle;
                }
            }
            else if (state == CoinbaseProState::downloading_live) {
                // Check if live data has stopped
                if (CoinbasePro_live->get_state() == CoinbaseProLiveState::idle) {
                    state = CoinbaseProState::idle;
                }
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

void CoinbasePro::interval_update_worker(void)
{
    while (state != CoinbaseProState::shutdown) {
        {
            auto interval_update_lock = std::unique_lock<std::mutex>{ interval_update_mutex };
            interval_update_condition.wait(interval_update_lock);
        }

        CoinbasePro_interval->update();
    }
}
