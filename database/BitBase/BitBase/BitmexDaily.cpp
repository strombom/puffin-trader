#include "BitmexDaily.h"


BitmexDailyState BitmexDaily::get_state(void)
{
    return state;
}

void BitmexDaily::start_download(void)
{
    state = BitmexDailyState::Downloading;
    download_thread = new boost::thread(&BitmexDaily::download, this);
}

void BitmexDaily::download(void)
{
    //logger.info("tick_data_last_timestamp %s", tick_data_last_timestamp.to_string().c_str());
    
    state = BitmexDailyState::Idle;
}
