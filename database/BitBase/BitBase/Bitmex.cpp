#include "Bitmex.h"
#include "Logger.h"
#include "DateTime.h"



Bitmex::Bitmex(Database& _database, DownloadManager& _download_manager)
{
    database = &_database;
    download_manager = &_download_manager;
    bitmex_daily = new BitmexDaily(_database, _download_manager);

    main_loop_thread = new boost::thread(&Bitmex::main_loop, this);
}

void Bitmex::shutdown(void)
{
    logger.info("Shutting down Bitmex client.");
    bitmex_daily->shutdown();
    state_mutex.lock();
    state = BitmexState::Shutdown;
    state_mutex.unlock();
    main_loop_thread->join();
}
 
void Bitmex::main_loop(void)
{
    while (state != BitmexState::Shutdown) {
        state_mutex.lock();

        if (state == BitmexState::Idle) {
            DateTime tick_data_last_timestamp = database->get_attribute("BITMEX", "BTCUSD", "tick_data_last_timestamp", bitmex_first_timestamp);
            if (tick_data_last_timestamp < DateTime::now() - TimeDelta::hours(24 + 1)) {
                // Last tick timestamp is more than 25 hours old, there should be a compressed daily archive available
                state = BitmexState::DownloadingDaily;
                bitmex_daily->start_download();
            }
        
        } else if (state == BitmexState::DownloadingDaily) {
            // Check if daily data is downloaded
            if (bitmex_daily->get_state() == BitmexDailyState::Idle) {
                state = BitmexState::Idle;
            }

        }

        state_mutex.unlock();

        boost::this_thread::sleep_for(boost::chrono::milliseconds(1000));
    }
}
