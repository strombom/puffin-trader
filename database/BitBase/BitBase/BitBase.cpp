
#include <iostream>

#include "Logger.h"
#include "Bitmex.h"
#include "Database.h"
#include "DownloadManager.h"


int main()
{
    sptrDownloadManager download_manager = DownloadManager::create();
    sptrDatabase database = Database::create("C:\\development\\github\\puffin-trader\\database\\data");

    Bitmex bitmex(database, download_manager);

    while (true) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }

    download_manager->shutdown();
    bitmex.shutdown();
}
