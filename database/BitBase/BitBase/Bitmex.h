#pragma once

#include <stdio.h>
#include <mutex>
#include <thread>
#include <future>

#include "Database.h"
#include "BitmexDaily.h"
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

private:
    sptrDatabase database;
    sptrDownloadManager download_manager;
    uptrBitmexDaily bitmex_daily;

    std::mutex state_mutex;
    bool thread_running;
    BitmexState state;

    std::future<void> main_loop_task;
};
