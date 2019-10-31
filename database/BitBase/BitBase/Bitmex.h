#pragma once

#include "Database.h"
#include "DownloadManager.h"

#include "boost/signals2.hpp"
#include "boost/thread.hpp"


class Bitmex
{
public:
    Bitmex(Database& _database, DownloadManager& _download_manager);

    void shutdown(void);
    void main_loop(void);

private:
    Database* database;
    DownloadManager* download_manager;

    bool running = true;

    boost::thread* main_loop_thread;
};

