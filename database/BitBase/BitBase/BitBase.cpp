
#include <iostream>

#include "Logger.h"
#include "Bitmex.h"
#include "Database.h"
#include "DownloadManager.h"


int main()
{
    DownloadManager download_manager;
    Database database("C:\\development\\github\\puffin-trader\\database\\data");
    Bitmex bitmex(database, download_manager);

    //system("pause");
    download_manager.shutdown();
    bitmex.shutdown();
}
