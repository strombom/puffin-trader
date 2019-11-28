
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
    state(BitmexDailyState::idle), tick_data_thread_running(true)
{
    tick_data_worker_thread = std::make_unique<std::thread>(&BitmexDaily::tick_data_worker, this);
}

BitmexDailyState BitmexDaily::get_state(void)
{
    //logger.info("BitmexDaily::get_state");
    return state;
}

void BitmexDaily::shutdown(void)
{
    logger.info("BitmexDaily::shutdown");
    tick_data_thread_running = false;
    download_manager->abort_client(downloader_client_id);
    state = BitmexDailyState::idle;

    if (tick_data_worker_thread->joinable()) {
        tick_data_worker_thread->join();
    }
}

void BitmexDaily::start_download(void)
{
    assert(state == BitmexDailyState::idle);
    logger.info("BitmexDaily::start_download");
    
    // We base daily data on BTCUSD timestamp, it has most activity and is of primary interest. Other symbols will be downloaded as well.
    const auto timestamp = database->get_attribute("BITMEX", "XBTUSD", "tick_data_last_timestamp", bitmex_first_timestamp);
    timestamp_next = date::floor<date::days>(timestamp);

    state = BitmexDailyState::downloading;

    for (int i = 0; i < active_downloads_max; ++i) {
        start_next();
    }
}

void BitmexDaily::download_done_callback(sptr_download_data_t payload)
{
    logger.info("BitmexDaily::download_done_callback start");

    boost::iostreams::array_source compressed(payload->data(), payload->size());
    boost::iostreams::filtering_streambuf<boost::iostreams::input> out;
    out.push(boost::iostreams::gzip_decompressor());
    out.push(compressed);

    std::stringstream decompressed;
    boost::iostreams::copy(out, decompressed);
    auto tick_data = parse_raw(decompressed);

    logger.info("BitmexDaily::download_done_callback parsing done");

    if (!tick_data) {
        logger.error("BitmexDaily::download_done_callback parsing error!");
        shutdown();
        return;
    }
    
    {
        std::scoped_lock slock(tick_data_mutex);
        tick_data_queue.push_back(std::move(tick_data));
    }
    tick_data_condition.notify_one();

    logger.info("BitmexDaily::download_done_callback start next");

    start_next();

    logger.info("BitmexDaily::download_done_callback end");
}

void BitmexDaily::start_next(void)
{
    logger.info("BitmexDaily::start_next start");

    time_point_us last_timestamp = system_clock_us_now() - date::days{ 1 };
    last_timestamp = date::floor<date::days>(last_timestamp);
    if (timestamp_next > last_timestamp) {
        state = BitmexDailyState::idle;
        logger.info("BitmexDaily::start_next last index");
        return;
    }
    
    std::string url = "https://s3-eu-west-1.amazonaws.com/public.bitmex.com/data/trade/";
    url += date::format("%Y%m%d", timestamp_next);
    url += ".csv.gz";
    download_manager->download(url, downloader_client_id, std::bind(&BitmexDaily::download_done_callback, this, std::placeholders::_1));
    timestamp_next += date::days{ 1 };

    logger.info("BitmexDaily::start_next end");
}

void BitmexDaily::tick_data_worker(void)
{
    while (tick_data_thread_running) {
        {
            std::unique_lock<std::mutex> tick_data_lock(tick_data_mutex);
            tick_data_condition.wait(tick_data_lock);
        }
        logger.info("BitmexDaily::tick_data_worker start");

        while (tick_data_thread_running) {
            uptrTickData tick_data;
            {
                std::scoped_lock slock(tick_data_mutex);
                if (!tick_data_queue.empty()) {
                    tick_data = std::move(tick_data_queue.front());
                    tick_data_queue.pop_front();
                }
                else {
                    break;
                }
            }

            const auto start = std::chrono::steady_clock::now();

            for (auto&& symbol_tick_data = tick_data->begin(); symbol_tick_data != tick_data->end(); ++symbol_tick_data) {
                const auto symbol = symbol_tick_data->first;
                auto data = std::move(symbol_tick_data->second);
                database->tick_data_extend(exchange_name, symbol, std::move(data), bitmex_first_timestamp);
            }

            const auto end = std::chrono::steady_clock::now();
            const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

            logger.info("BitmexDaily::tick_data_worker tick_data appended to database (%d ms)", elapsed);
        }
        logger.info("BitmexDaily::tick_data_worker end");
    }
}

BitmexDaily::uptrTickData BitmexDaily::parse_raw(const std::stringstream& raw_data)
{
    auto tick_data = std::make_unique<TickData>();

    logger.info("BitmexDaily::parse_raw");
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
            return nullptr;
        }
        if ((*tick_data)[symbol] == nullptr) {
            (*tick_data)[symbol] = std::make_unique<DatabaseTicks>();
        }
        (*tick_data)[symbol]->append(timestamp, price, volume, buy);
    }

    return tick_data;
}
