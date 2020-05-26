#include "pch.h"

#include "Bitmex.h"
#include "Logger.h"
#include "DateTime.h"


Bitmex::Bitmex(sptrDatabase database, sptrDownloadManager download_manager) :
    database(database), download_manager(download_manager), state(BitmexState::idle), thread_running(true)
{
    bitmex_daily = std::make_unique<BitmexDaily>(database, download_manager, std::bind(&Bitmex::tick_data_updated_callback, this));
    bitmex_interim = std::make_unique<BitmexInterim>(database, std::bind(&Bitmex::tick_data_updated_callback, this));
    bitmex_live = std::make_unique<BitmexLive>(database, std::bind(&Bitmex::tick_data_updated_callback, this));
    bitmex_interval = std::make_unique<BitmexInterval>(database);

    main_loop_thread = std::make_unique<std::thread>(&Bitmex::main_loop, this);
    interval_update_thread = std::make_unique<std::thread>(&Bitmex::interval_update_worker, this);
}

void Bitmex::shutdown(void)
{
    logger.info("Bitmex::shutdown");
    {
        auto slock = std::scoped_lock{ state_mutex };
        state = BitmexState::shutdown;
    }
    logger.info("Bitmex::shutdown state = shutdown");
    bitmex_daily->shutdown();
    bitmex_interval->shutdown();

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

void Bitmex::tick_data_updated_callback(void)
{
    interval_update_condition.notify_one();
}

void Bitmex::main_loop(void)
{
    while (state != BitmexState::shutdown) {
        {
            auto slock = std::scoped_lock{ state_mutex };

            if (state == BitmexState::idle) {
                auto tick_data_last_timestamp = database->get_attribute("BITMEX", "XBTUSD", "tick_data_last_timestamp", BitBase::Bitmex::first_timestamp);
                if (tick_data_last_timestamp < std::chrono::system_clock::now() - std::chrono::hours{ 24 + 6 }) {
                    // Last tick timestamp is more than 24 + 6 hours old, there should be a compressed daily archive available
                    state = BitmexState::downloading_daily;
                    bitmex_daily->start();
                }
                else if (tick_data_last_timestamp < std::chrono::system_clock::now() - std::chrono::minutes{ 1 }) {
                    state = BitmexState::downloading_interim;
                    bitmex_interim->start();
                }
                else {
                    state = BitmexState::downloading_live;
                    bitmex_live->start();
                }

            } else if (state == BitmexState::downloading_daily) {
                // Check if daily data is downloaded
                if (bitmex_daily->get_state() == BitmexDailyState::idle) {
                    state = BitmexState::idle;
                }
            }
            else if (state == BitmexState::downloading_interim) {
                // Check if interim data is up to date
                if (bitmex_interim->get_state() == BitmexInterimState::idle) {
                    state = BitmexState::idle;
                }
            }
            else if (state == BitmexState::downloading_live) {
                // Check if live data has stopped
                if (bitmex_live->get_state() == BitmexLiveState::idle) {
                    state = BitmexState::idle;
                }
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
}

void Bitmex::interval_update_worker(void)
{
    while (state != BitmexState::shutdown) {
        {
            auto interval_update_lock = std::unique_lock<std::mutex>{ interval_update_mutex };
            interval_update_condition.wait(interval_update_lock);
        }

        bitmex_interval->update();
    }
}
