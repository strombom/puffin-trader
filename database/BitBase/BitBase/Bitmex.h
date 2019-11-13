#pragma once

#include <stdio.h>
#include <mutex>
#include <thread>

#include "Database.h"
#include "BitmexDaily.h"
#include "BitmexConstants.h"
#include "DownloadManager.h"


enum class BitmexState { 
    Idle, 
    DownloadingDaily,
    Shutdown
};

class Bitmex
{
public:
    Bitmex(Database& _database, DownloadManager& _download_manager);

    void shutdown(void);
    void main_loop(void);

private:
    Database* database;
    DownloadManager* download_manager;
    BitmexDaily* bitmex_daily;

    std::mutex state_mutex;
    bool thread_running = true;
    BitmexState state = BitmexState::Idle;

    std::thread* main_loop_thread;
};

