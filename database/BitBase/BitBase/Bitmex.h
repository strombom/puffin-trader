#pragma once

#include <stdio.h>
#include <mutex>
#include <thread>
#include <future>

#include "Database.h"
#include "BitmexDaily.h"
#include "BitmexInterval.h"
#include "BitmexConstants.h"
#include "DownloadManager.h"


enum class BitmexState { 
    idle, 
    downloading_daily,
    shutdown
};

class Bitmex
{
public:
    Bitmex(sptrDatabase database, sptrDownloadManager download_manager);

    void shutdown(void);
    void main_loop(void);
    void tick_data_updated_callback(void);

private:
    sptrDatabase database;
    sptrDownloadManager download_manager;
    uptrBitmexDaily bitmex_daily;
    uptrBitmexInterval bitmex_interval;

    std::mutex state_mutex;
    bool thread_running;
    BitmexState state;

    std::unique_ptr<std::thread> main_loop_thread;
    //std::future<void> main_loop_task;
};
