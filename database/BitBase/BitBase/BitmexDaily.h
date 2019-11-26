#pragma once

#include "DateTime.h"
#include "Database.h"
#include "DownloadManager.h"
#include "BitmexConstants.h"

#include <mutex>


enum class BitmexDailyState {
    idle,
    downloading
};

class BitmexDaily
{
public:
    BitmexDaily(sptrDatabase database, sptrDownloadManager download_manager);

    BitmexDailyState get_state(void);
    void start_download(void);
    void shutdown(void);

private:
    using TickData = std::map<std::string, DatabaseTicks>;
    using sptrTickData = std::shared_ptr<TickData>;

    inline static const std::string exchange_name = "BITMEX";
    inline static const std::string downloader_client_id = "bitmex_daily";
    static const int active_downloads_max = 5;

    std::atomic<BitmexDailyState> state;
    std::mutex state_mutex;

    sptrDatabase database;
    sptrDownloadManager download_manager;
    time_point_us downloading_first;
    time_point_us downloading_last;
    int active_downloads_count;

    bool start_next(void);
    void download_done_callback(sptr_download_data_t payload);

    bool parse_raw(const std::stringstream& raw_data, sptrTickData tick_data);
};

using sptrBitmexDaily = std::shared_ptr<BitmexDaily>;
