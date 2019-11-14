
#include "BitmexDaily.h"
#include "Logger.h"


BitmexDaily::BitmexDaily(sptrDatabase database, sptrDownloadManager download_manager) :
    database(database), download_manager(download_manager), 
    state(BitmexDailyState::idle), active_downloads_count(0)
{

}

BitmexDailyState BitmexDaily::get_state(void)
{
    std::scoped_lock lock(state_mutex);
    return state;
}

void BitmexDaily::shutdown(void)
{
    download_manager->abort(client_id);
}

void BitmexDaily::start_download(void)
{
    std::scoped_lock lock(state_mutex);

    active_downloads_count = 0;
    downloading_first = database->get_attribute("BITMEX", "BTCUSD", "tick_data_last_timestamp", bitmex_first_timestamp);
    downloading_first.set_time(0, 0, 0);
    downloading_last = downloading_first;
    state = BitmexDailyState::downloading;

    while (start_next()); // Starting as many downloads as possible.
 }

bool BitmexDaily::start_next(void)
{
    if (active_downloads_count == active_downloads_max) {
        return false;
    }

    DateTime last_timestamp = DateTime::now() - TimeDelta::days(1);
    last_timestamp.set_time(0, 0, 0);
    if (downloading_last > last_timestamp) {
        state = BitmexDailyState::idle;
        return false;
    }
    
    std::string url = "https://s3-eu-west-1.amazonaws.com/public.bitmex.com/data/trade/";
    url += downloading_last.to_string("%Y%m%d");
    url += ".csv.gz";

    download_manager->download(url, client_id, downloading_last.to_string(), std::bind(&BitmexDaily::download_done_callback, this, _1, _2));

    downloading_last += TimeDelta::days(1);
    active_downloads_count += 1;

    return true;
}

void BitmexDaily::download_done_callback(std::string datestring, payload_t payload)
{
    logger.info("BitmexDaily download done (%s).", datestring.c_str());
    active_downloads_count--;
    if (active_downloads_count == 0) {

    } else {
        start_next();
    }
}
