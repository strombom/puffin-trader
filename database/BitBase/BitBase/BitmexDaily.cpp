
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
    std::map<std::string, DatabaseTicks> tables;
    
    const boost::regex linesregx("\\n");
    std::string indata = raw_data.str();
    boost::sregex_token_iterator row_it(indata.begin(), indata.end(), linesregx, -1);
    boost::sregex_token_iterator row_end;

    ++row_it; // Skip table headers
    while (row_it != row_end) {
        std::string row = row_it->str();
        ++row_it;

        const boost::regex fieldsregx(",");
        boost::sregex_token_iterator col_it(row.begin(), row.end(), fieldsregx, -1);
        boost::sregex_token_iterator col_end;

        DateTime timestamp;
        std::string symbol;
        float price;
        float volume;
        bool buy;
        bool valid = false;

        int idx = 0;
        while (col_it != col_end) {
            std::string token = col_it->str();
            ++col_it;
            if (idx == 0) {
                timestamp = DateTime(token, "%Y-%m-%dD%H:%M:%s");
            }
            else if (idx == 1) {
                symbol = token;
            }
            else if (idx == 2) {
                buy = (token == "Buy");
            }
            else if (idx == 3) {
                volume = std::stof(token);
            }
            else if (idx == 4) {
                price = std::stof(token);
                valid = true;
                break;
            }
            ++idx;
        }

        if (!valid) {
            return;
        }
        tables[symbol].append(timestamp, price, volume, buy);
    }

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
