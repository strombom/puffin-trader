#pragma once

#include <stdio.h>
#include "boost/signals2.hpp"
#include "boost/thread.hpp"

#include "Database.h"
#include "BitmexDaily.h"
#include "BitmexConstants.h"
#include "DownloadManager.h"


enum class BitmexState { 
    Idle, 
    DownloadingDaily
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
    BitmexDaily bitmex_daily;

    bool thread_running = true;
    BitmexState state = BitmexState::Idle;

    boost::thread* main_loop_thread;
};

