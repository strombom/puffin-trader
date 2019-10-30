#pragma once

#include "Database.h"
#include "DownloadManager.h"


class Bitmex
{
public:
    Bitmex(Database& _database, DownloadManager& _download_manager);

private:
    Database* database;
    DownloadManager* download_manager;
};

