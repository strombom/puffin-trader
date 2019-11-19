
#include "BitmexDaily.h"
#include "Logger.h"

#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filter/zlib.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/device/array.hpp>
#include <boost/iostreams/copy.hpp>
#include <boost/regex.hpp>
#include <string>


BitmexDaily::BitmexDaily(sptrDatabase database, sptrDownloadManager download_manager) :
    database(database), download_manager(download_manager), 
    state(BitmexDailyState::idle), active_downloads_count(0)
{

}

BitmexDailyState BitmexDaily::get_state(void)
{
    std::scoped_lock lock(state_mutex);

    return state;
}

void BitmexDaily::shutdown(void)
{
    std::scoped_lock lock(state_mutex);

    download_manager->abort_client(downloader_client_id);
    state = BitmexDailyState::idle;
}

void BitmexDaily::start_download(void)
{
    std::scoped_lock lock(state_mutex);

    active_downloads_count = 0;
    downloading_first = database->get_attribute("BITMEX", "BTCUSD", "tick_data_last_timestamp", bitmex_first_timestamp);
    downloading_first.set_time(0, 0, 0);
    downloading_last = downloading_first;
    state = BitmexDailyState::downloading;

    while (start_next()); // Starting as many downloads as possible.
 }

void BitmexDaily::parse_raw(const std::stringstream& raw_data)
{
    typedef std::vector<std::string> Row;

    std::vector<Row> result;

    
    // iterator splits data to lines
    //std::string data = raw_data.str();

    const boost::regex linesregx("\\n");
    const boost::regex fieldsregx(",");


    std::string indata = raw_data.str();
    //const char* data = raw_data.str().c_str();
    //const unsigned int length = (unsigned int) raw_data.str().length();

    boost::sregex_token_iterator li(indata.begin(), indata.end(), linesregx, -1);
    boost::sregex_token_iterator end;

    while (li != end) {
        std::string line = li->str();
        ++li;

        // Split line to tokens
        boost::sregex_token_iterator ti(line.begin(), line.end(), fieldsregx, -1);
        boost::sregex_token_iterator end2;

        std::vector<std::string> row;
        while (ti != end2) {
            std::string token = ti->str();
            ++ti;
            row.push_back(token);
        }
        if (line.back() == ',') {
            // last character was a separator
            row.push_back("");
        }
        result.push_back(row);
    }




    std::string s;
    s = raw_data.str();
    logger.info("payload %s", s.c_str());


    //std::vector<DateTime>>

    /*


    xt::xarray<double> arr1
    { {1.0, 2.0, 3.0},
     {2.0, 5.0, 7.0},
     {2.0, 5.0, 7.0} };

    xt::xarray<double> arr2
    { 5.0, 6.0, 7.0 };

    xt::xarray<double> res = xt::view(arr1, 1) + arr2;

    std::stringstream ss;
    std::string s2;
    ss << res;

    logger.info("res %s", ss.str().c_str());
    */
}


void BitmexDaily::download_done_callback(std::string datestring, sptr_download_data_t payload)
{
    boost::iostreams::array_source compressed(payload->data(), payload->size());
    boost::iostreams::filtering_streambuf<boost::iostreams::input> out;
    out.push(boost::iostreams::gzip_decompressor());
    out.push(compressed);

    std::stringstream decompressed;
    boost::iostreams::copy(out, decompressed);
    parse_raw(decompressed);
    
    std::scoped_lock lock(state_mutex);
    logger.info("BitmexDaily download done (%s)", datestring.c_str());
    active_downloads_count--;
    if (active_downloads_count == 0) {
        state = BitmexDailyState::idle;
    } else {
        start_next();
    }
}

bool BitmexDaily::start_next(void)
{
    if (active_downloads_count == active_downloads_max) {
        return false;
    }

    DateTime last_timestamp = DateTime::now() - TimeDelta::days(1);
    last_timestamp.set_time(0, 0, 0);
    if (downloading_last > last_timestamp) {
        state = BitmexDailyState::idle;
        return false;
    }
    
    std::string url = "https://s3-eu-west-1.amazonaws.com/public.bitmex.com/data/trade/";
    url += downloading_last.to_string("%Y%m%d");
    url += ".csv.gz";

    download_manager->download(url, downloader_client_id, downloading_last.to_string_date(), std::bind(&BitmexDaily::download_done_callback, this, std::placeholders::_1, std::placeholders::_2));

    downloading_last += TimeDelta::days(1);
    active_downloads_count += 1;

    return true;
}
