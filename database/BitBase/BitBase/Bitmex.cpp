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
    running = false;
    main_loop_thread->join();
}

void Bitmex::main_loop(void)
{
    while (running) {

        DateTime tick_data_last_timestamp = database->get_attribute("BITMEX", "BTCUSD" ,"tick_data_last_timestamp", dataset_first_timestamp);
        logger.info("tick_data_last_timestamp %s", tick_data_last_timestamp.to_string().c_str());

        DateTime yesterday = DateTime(); 
        logger.info("now %s", yesterday.to_string().c_str());
        yesterday = yesterday - TimeDelta(Duration::days(1), Duration::hours(1));
        logger.info("yesterday %s", yesterday.to_string().c_str());
        /*now.set_hour(0);
        now.set_minute(0);
        now.set_second(0);*/


        boost::this_thread::sleep_for(boost::chrono::milliseconds(1000));
    }
}
