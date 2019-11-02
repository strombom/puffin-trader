#include "Bitmex.h"
#include "Logger.h"
#include "DateTime.h"


Bitmex::Bitmex(Database& _database, DownloadManager& _download_manager)
{
    database = &_database;
    download_manager = &_download_manager;

    logger.info("hello Bitmex");
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

        DateTime tíck_data_last_timestamp = database->get_attribute_date("bitmex", "tíck_data_last_timestamp");

        logger.info("tick_data_last_timestamp %s", tíck_data_last_timestamp.to_string());
        //break;

        boost::this_thread::sleep_for(boost::chrono::milliseconds(1000));
    }
}
