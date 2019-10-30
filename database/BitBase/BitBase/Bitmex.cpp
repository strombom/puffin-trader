#include "Bitmex.h"
#include "Logger.h"


Bitmex::Bitmex(Database& _database, DownloadManager& _download_manager)
{
    database = &_database;
    download_manager = &_download_manager;
}
