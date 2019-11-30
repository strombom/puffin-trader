

#include "Bitmex.h"
#include "Logger.h"
#include "DateTime.h"


Bitmex::Bitmex(sptrDatabase database, sptrDownloadManager download_manager) :
    database(database), download_manager(download_manager), state(BitmexState::idle), thread_running(true)
{
    bitmex_daily = std::make_unique<BitmexDaily>(database, download_manager);
    //main_loop_task = std::async(&Bitmex::main_loop, this);

    main_loop_thread = std::make_unique<std::thread>(&Bitmex::main_loop, this);
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

    try {
        main_loop_thread->join();
    }
    catch (...) {}
}
 
void Bitmex::main_loop(void)
{
    while (state != BitmexState::shutdown) {
        {
            auto slock = std::scoped_lock{ state_mutex };

            if (state == BitmexState::idle) {
                auto tick_data_last_timestamp = database->get_attribute("BITMEX", "BTCUSD", "tick_data_last_timestamp", bitmex_first_timestamp);
                if (tick_data_last_timestamp < std::chrono::system_clock::now() - std::chrono::hours{ 24 + 1 }) {
                    // Last tick timestamp is more than 25 hours old, there should be a compressed daily archive available
                    state = BitmexState::downloading_daily;
                    bitmex_daily->start_download();
                }

            } else if (state == BitmexState::downloading_daily) {
                // Check if daily data is downloaded
                if (bitmex_daily->get_state() == BitmexDailyState::idle) {
                    state = BitmexState::idle;
                }

            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
}
