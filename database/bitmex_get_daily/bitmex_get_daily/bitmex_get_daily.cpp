// bitmex_get_daily.cpp
//

#include "DownloadManager.h"

#include "H5Cpp.h"


int main()
{
    DownloadManager download_manager;

    download_manager.download(boost::gregorian::date(2017, 05, 21));
    download_manager.download(boost::gregorian::date(2017, 05, 22));
    download_manager.download(boost::gregorian::date(2017, 05, 23));
    download_manager.download(boost::gregorian::date(2017, 05, 25));

    download_manager.join();

    return 0;
}
