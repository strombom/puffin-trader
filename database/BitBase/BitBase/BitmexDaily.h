#pragma once

#include "boost/thread.hpp"

#include "DateTime.h"
#include "Database.h"
#include "DownloadManager.h"
#include "BitmexConstants.h"


enum class BitmexDailyState {
    Idle,
    Downloading
};

class BitmexDaily
{
public:
    BitmexDaily(Database& _database, DownloadManager& _download_manager);

    BitmexDailyState get_state(void);
    void start_download(void);
    void shutdown(void);

private:
    BitmexDailyState state = BitmexDailyState::Idle;
    boost::mutex state_mutex;

    Database* database;
    DownloadManager* download_manager;
    DateTime downloading_first;
    DateTime downloading_last;

    static const int active_downloads_max = 5;
    int active_downloads_count = 0;
    boost::thread* download_thread;
    void download(void);
    bool start_next(void);
    void download_done_callback(std::string datestring, std::shared_ptr<std::vector<std::byte>>);

};
