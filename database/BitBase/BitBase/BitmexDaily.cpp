
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

    // We base daily data on BTCUSD timestamp, it has most activity and is of primary interest. Other symbols will be downloaded as well.
    downloading_first = database->get_attribute("BITMEX", "XBTUSD", "tick_data_last_timestamp", bitmex_first_timestamp);
    downloading_first = date::floor<date::days>(downloading_first);
    downloading_last = downloading_first;
    state = BitmexDailyState::downloading;

    while (start_next()); // Starting as many downloads as possible.
 }

bool BitmexDaily::parse_raw(const std::stringstream& raw_data, sptrTickData tick_data)
{
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

        time_point_us timestamp;
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
                std::istringstream ss{ token };
                ss >> date::parse("%FD%T", timestamp);
                if (ss.fail()) {
                    break;
                }
            }
            else if (idx == 1) {
                symbol = token;
            }
            else if (idx == 2) {
                if (token == "Buy") {
                    buy = true;
                }
                else if (token == "Sell") {
                    buy = false;
                }
                else if (token == "") {
                    buy = true;
                }
                else {
                    break;
                }
            }
            else if (idx == 3) {
                try {
                    volume = std::stof(token);
                }
                catch (...) {
                    break; // Invalid volume format
                }
            }
            else if (idx == 4) {
                try {
                    price = std::stof(token);
                    valid = true;
                    break;
                }
                catch (...) {
                    break; // Invalid price format
                }
            }
            ++idx;
        }

        if (!valid) {
            return false;
        }
        (*tick_data)[symbol].append(timestamp, price, volume, buy);
    }

    return true;
}

void BitmexDaily::download_done_callback(std::string datestring, sptr_download_data_t payload)
{
    std::scoped_lock lock(state_mutex);

    boost::iostreams::array_source compressed(payload->data(), payload->size());
    boost::iostreams::filtering_streambuf<boost::iostreams::input> out;
    out.push(boost::iostreams::gzip_decompressor());
    out.push(compressed);

    std::stringstream decompressed;
    boost::iostreams::copy(out, decompressed);
    auto tick_data = std::make_shared<TickData>();
    bool valid = parse_raw(decompressed, tick_data);

    if (!valid) {
        shutdown();
        return;
    }

    for (auto&& symbol_keyval = tick_data->begin(); symbol_keyval != tick_data->end(); ++symbol_keyval) {
        auto symbol = symbol_keyval->first;
        auto data = std::make_shared<DatabaseTicks>(symbol_keyval->second);
        database->tick_data_extend(exchange_name, symbol, data, bitmex_first_timestamp);
    }

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

    time_point_us last_timestamp = system_clock_us_now() - date::days{ 1 };
    last_timestamp = date::floor<date::days>(last_timestamp);
    if (downloading_last > last_timestamp) {
        state = BitmexDailyState::idle;
        return false;
    }
    
    std::string url = "https://s3-eu-west-1.amazonaws.com/public.bitmex.com/data/trade/";
    url += date::format("%Y%m%d", downloading_last);
    url += ".csv.gz";

    download_manager->download(url, downloader_client_id, date::format("%Y-%m-%d", downloading_last), std::bind(&BitmexDaily::download_done_callback, this, std::placeholders::_1, std::placeholders::_2));

    downloading_last += date::days{ 1 };
    active_downloads_count += 1;

    return true;
}
