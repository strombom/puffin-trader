#pragma once

#include "DateTime.h"
#include "Database.h"
#include "DownloadManager.h"
#include "BitBaseConstants.h"

#include <mutex>
#include <string>

using tick_data_updated_callback_t = std::function<void(void)>;

enum class BitmexDailyState {
    idle,
    downloading
};

class BitmexDaily
{
public:
    BitmexDaily(sptrDatabase database, sptrDownloadManager download_manager, tick_data_updated_callback_t tick_data_updated_callback);

    BitmexDailyState get_state(void);
    void start_download(void);
    void shutdown(void);

private:
    using TickData = std::map<std::string, uptrDatabaseTicks>;
    using uptrTickData = std::unique_ptr<TickData>;
    
    std::mutex start_download_mutex;
    std::atomic<BitmexDailyState> state;

    sptrDatabase database;
    sptrDownloadManager download_manager;
    time_point_us timestamp_next;

    std::mutex tick_data_mutex;
    std::unique_ptr<std::thread> tick_data_worker_thread;
    std::deque<uptrTickData> tick_data_queue;
    std::condition_variable tick_data_condition;
    std::atomic_bool tick_data_thread_running;
    tick_data_updated_callback_t tick_data_updated_callback;

    void start_next_download(void);
    void download_done_callback(sptr_download_data_t payload);
    void update_symbol_names(const std::unordered_set<std::string>& new_symbol_names);
    void tick_data_worker(void);
    uptrTickData parse_raw(const std::stringstream& raw_data);
};

using uptrBitmexDaily = std::unique_ptr<BitmexDaily>;
