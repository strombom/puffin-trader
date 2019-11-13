
#include "BitmexDaily.h"
#include "Logger.h"

#include <cstddef>
//#include <stdio.h>
//#include "boost/bind.hpp"

BitmexDaily::BitmexDaily(Database& _database, DownloadManager& _download_manager)
{
    database = &_database;
    download_manager = &_download_manager;
}

BitmexDailyState BitmexDaily::get_state(void)
{
    std::scoped_lock lock(state_mutex);
    return state;
}

void BitmexDaily::shutdown(void)
{

}

void BitmexDaily::start_download(void)
{
    std::scoped_lock lock(state_mutex);

    active_downloads_count = 0;
    downloading_first = database->get_attribute("BITMEX", "BTCUSD", "tick_data_last_timestamp", bitmex_first_timestamp);
    downloading_first.set_time(0, 0, 0);
    downloading_last = downloading_first;
    state = BitmexDailyState::Downloading;

    while (start_next()) {
        // Starting as many downloads as possible.
    }

    //download_thread = new boost::thread(&BitmexDaily::download, this);
}

bool BitmexDaily::start_next(void)
{
    if (active_downloads_count == active_downloads_max) {
        return false;
    }

    DateTime last_timestamp = DateTime::now() - TimeDelta::days(1);
    last_timestamp.set_time(0, 0, 0);
    if (downloading_last > last_timestamp) {
        return false;
    }
    
    std::string url = "https://s3-eu-west-1.amazonaws.com/public.bitmex.com/data/trade/";
    url += downloading_last.to_string("%Y%m%d");
    url += ".csv.gz";

    download_manager->download(url, "bitmex_daily", downloading_last.to_string(), std::bind(&BitmexDaily::download_done_callback, this, _1, _2));

    downloading_last += TimeDelta::days(1);
    active_downloads_count += 1;

    return true;
}

void BitmexDaily::download_done_callback(std::string datestring, std::shared_ptr<std::vector<std::byte>> payload)
{
    logger.info("BitmexDaily download done (%s).", datestring.c_str());
}
/*
void BitmexDaily::download(void)
{
    boost::mutex::scoped_lock lock(state_mutex);

    //logger.info("tick_data_last_timestamp %s", tick_data_last_timestamp.to_string().c_str());
    
    state = BitmexDailyState::Idle;
}
*/