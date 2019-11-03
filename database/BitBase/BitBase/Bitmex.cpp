#include "Bitmex.h"
#include "Logger.h"
#include "DateTime.h"

static const DateTime dataset_first_timestamp(2017, 01, 01, 00, 00, 00.0);


Bitmex::Bitmex(Database& _database, DownloadManager& _download_manager)
{
    database = &_database;
    download_manager = &_download_manager;

    main_loop_thread = new boost::thread(&Bitmex::main_loop, this);
}

void Bitmex::shutdown(void)
{
    logger.info("Shutting down Bitmex client.");
    thread_running = false;
    main_loop_thread->join();
}

void Bitmex::main_loop(void)
{
    while (thread_running) {

        if (state == BitmexState::Idle) {
            DateTime tick_data_last_timestamp = database->get_attribute("BITMEX", "BTCUSD", "tick_data_last_timestamp", dataset_first_timestamp);
            if (tick_data_last_timestamp < DateTime::now() - TimeDelta::hours(48 + 1)) {
                // Last tick timestamp is more than 48 hours old, there should be a compressed daily archive available
                state = BitmexState::DownloadingDaily;
                //logger.info("tick_data_last_timestamp %s", tick_data_last_timestamp.to_string().c_str());
            }
        
        } else if (state == BitmexState::DownloadingDaily) {


        }

        boost::this_thread::sleep_for(boost::chrono::milliseconds(1000));
    }
}
