#include "Bitmex.h"
#include "Logger.h"
#include "DateTime.h"


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

        DateTime last_daily = database->get_attribute_date("bitmex", "last_daily");



        boost::this_thread::sleep_for(boost::chrono::milliseconds(100));
    }
}
