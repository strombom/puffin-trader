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
    using TickData = std::map<std::string, std::unique_ptr<DatabaseTicks>>;
    using uptrTickData = std::unique_ptr<TickData>;

    inline static const std::string exchange_name = "BITMEX";
    inline static const std::string downloader_client_id = "bitmex_daily";
    static const int active_downloads_max = 5;

    std::atomic<BitmexDailyState> state;

    sptrDatabase database;
    sptrDownloadManager download_manager;
    time_point_us timestamp_next;

    std::unique_ptr<std::thread> tick_data_worker_thread;
    std::deque<uptrTickData> tick_data_queue;
    std::mutex tick_data_mutex;
    std::condition_variable tick_data_condition;
    std::atomic<bool> tick_data_thread_running;

    void start_next_download(void);
    void download_done_callback(sptr_download_data_t payload);
    void tick_data_worker(void);
    uptrTickData parse_raw(const std::stringstream& raw_data);
};

using uptrBitmexDaily = std::unique_ptr<BitmexDaily>;
