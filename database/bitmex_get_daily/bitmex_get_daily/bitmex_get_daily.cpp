// bitmex_get_daily.cpp
//

#pragma warning (disable : 26444)

#include "DownloadManager.h"
#include "BitmexDaily.h"

#include "boost/date_time/gregorian/gregorian.hpp"


std::string date_to_string(boost::gregorian::date date)
{
    std::stringstream string;
    string.imbue(std::locale(std::cout.getloc(), new boost::date_time::date_facet < boost::gregorian::date, char>("%Y%m%d")));
    string << date;
    return string.str();
}

int main()
{
    BitmexDaily bitmex_daily("BTCUSD");

    /*
    DownloadManager download_manager("https://s3-eu-west-1.amazonaws.com/public.bitmex.com/data/trade/", ".csv.gz");

    download_manager.download(date_to_string(boost::gregorian::date(2017, 05, 21)));
    download_manager.download(date_to_string(boost::gregorian::date(2017, 05, 22)));
    download_manager.download(date_to_string(boost::gregorian::date(2017, 05, 23)));
    download_manager.download(date_to_string(boost::gregorian::date(2017, 05, 24)));
    download_manager.download(date_to_string(boost::gregorian::date(2017, 05, 25)));
    download_manager.download(date_to_string(boost::gregorian::date(2017, 05, 26)));
    download_manager.download(date_to_string(boost::gregorian::date(2017, 05, 27)));
    download_manager.download(date_to_string(boost::gregorian::date(2017, 05, 28)));

    download_manager.join();
    */

    return 0;
}
