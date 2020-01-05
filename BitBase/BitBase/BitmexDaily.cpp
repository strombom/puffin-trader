
#include "BitBaseConstants.h"
#include "BitmexDaily.h"
#include "DateTime.h"
#include "Logger.h"

#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/copy.hpp>
#include <regex>
#include <string>
#include <array>


BitmexDaily::BitmexDaily(sptrDatabase database, sptrDownloadManager download_manager, tick_data_updated_callback_t tick_data_updated_callback) :
    database(database), download_manager(download_manager), tick_data_updated_callback(tick_data_updated_callback),
    state(BitmexDailyState::idle), tick_data_thread_running(true)
{
    // We base daily data on BTCUSD timestamp, it has most activity and is of primary interest. Other symbols will be downloaded as well.
    const auto timestamp = database->get_attribute(BitBase::Bitmex::exchange_name, "XBTUSD", "tick_data_last_timestamp", BitBase::Bitmex::first_timestamp);
    timestamp_next = date::floor<date::days>(timestamp) + date::days{ 1 };

    tick_data_worker_thread = std::make_unique<std::thread>(&BitmexDaily::tick_data_worker, this);
}

BitmexDailyState BitmexDaily::get_state(void)
{
    return state;
}

void BitmexDaily::shutdown(void)
{
    logger.info("BitmexDaily::shutdown");

    {
        // Will not start new downloads after this section
        auto slock = std::scoped_lock{ start_download_mutex };
        state = BitmexDailyState::idle;
    }

    tick_data_thread_running = false;
    tick_data_condition.notify_all();

    download_manager->abort_client(BitBase::Bitmex::Daily::downloader_client_id);

    try {
        tick_data_worker_thread->join();
    }
    catch (...) {}
}

void BitmexDaily::start_download(void)
{
    assert(state == BitmexDailyState::idle);
    logger.info("BitmexDaily::start_download");
    

    state = BitmexDailyState::downloading;

    for (int i = 0; i < BitBase::Bitmex::Daily::active_downloads_max; ++i) {
        start_next_download();
    }
}

void BitmexDaily::download_done_callback(sptr_download_data_t payload)
{
    auto compressed = boost::iostreams::array_source{ payload->data(), payload->size() };
    auto out = boost::iostreams::filtering_streambuf<boost::iostreams::input>{};
    out.push(boost::iostreams::gzip_decompressor{});
    out.push(compressed);

    auto decompressed = std::stringstream{};
    boost::iostreams::copy(out, decompressed);
    auto tick_data = parse_raw(decompressed);

    if (!tick_data) {
        logger.error("BitmexDaily::download_done_callback parsing error!");
        shutdown();
        return;
    }
    
    {
        auto slock = std::scoped_lock{ tick_data_mutex };
        tick_data_queue.push_back(std::move(tick_data));
    }
    tick_data_condition.notify_one();

    start_next_download();
}

void BitmexDaily::start_next_download(void)
{
    if (state == BitmexDailyState::idle) {
        //logger.info("BitmexDaily::start_next end (idle)");
        return;
    }

    auto last_timestamp = date::floor<date::days>(system_clock_us_now() - date::days{ 1 });
    if (timestamp_next > last_timestamp) {
        state = BitmexDailyState::idle;
        //logger.info("BitmexDaily::start_next end (last index)");
        return;
    }

    auto url = std::string{ BitBase::Bitmex::Daily::base_url_start } + date::format(BitBase::Bitmex::Daily::url_date_format, timestamp_next) + std::string{ BitBase::Bitmex::Daily::base_url_end };
    download_manager->download(url, BitBase::Bitmex::Daily::downloader_client_id, std::bind(&BitmexDaily::download_done_callback, this, std::placeholders::_1));
    timestamp_next += date::days{ 1 };
}

void BitmexDaily::update_symbol_names(const std::unordered_set<std::string>& new_symbol_names)
{
    auto symbol_names = database->get_attribute(BitBase::Bitmex::exchange_name, "symbols", std::unordered_set<std::string>{});
    for (auto&& symbol_name : new_symbol_names) {
        symbol_names.insert(symbol_name);
    }
    database->set_attribute(BitBase::Bitmex::exchange_name, "symbols", symbol_names);
}

void BitmexDaily::tick_data_worker(void)
{
    while (tick_data_thread_running) {
        {
            auto tick_data_lock = std::unique_lock<std::mutex>{ tick_data_mutex };
            tick_data_condition.wait(tick_data_lock);
        }

        while (tick_data_thread_running) {
            auto tick_data = uptrTickData{};
            {
                auto slock = std::scoped_lock{ tick_data_mutex };
                if (!tick_data_queue.empty()) {
                    tick_data = std::move(tick_data_queue.front());
                    tick_data_queue.pop_front();
                }
                else {
                    break;
                }
            }

            auto timer = Timer{};
            auto symbol_names = std::unordered_set<std::string>{};
            for (auto&& symbol_tick_data = tick_data->begin(); symbol_tick_data != tick_data->end(); ++symbol_tick_data) {
                const auto symbol_name = symbol_tick_data->first;
                symbol_names.insert(symbol_name);
                database->extend_tick_data(BitBase::Bitmex::exchange_name, symbol_name, std::move(symbol_tick_data->second), BitBase::Bitmex::first_timestamp);
            }
            update_symbol_names(symbol_names);
            tick_data_updated_callback();

            logger.info("BitmexDaily::tick_data_worker tick_data appended to database (%d ms)", timer.elapsed().count()/1000);
        }
    }
    logger.info("BitmexDaily::tick_data_worker exit");
}

BitmexDaily::uptrTickData BitmexDaily::parse_raw(const std::stringstream& raw_data)
{
    auto timer = Timer{};

    auto tick_data = std::make_unique<TickData>();

    const auto linesregx = std::regex{ "\\n" };
    const auto indata = std::string{ raw_data.str() };
    auto row_it = std::sregex_token_iterator{ indata.begin(), indata.end(), linesregx, -1 };
    auto row_end = std::sregex_token_iterator{};

    ++row_it; // Skip table headers
    while (row_it != row_end) {
        const auto row = std::string{ row_it->str() };
        ++row_it;

        if (row.length() < 40) {
            return nullptr;
        }

        auto timestamp = time_point_us{};
        auto ss = std::istringstream{ row.substr(0, 29) };
        ss >> date::parse("%FD%T", timestamp);
        if (ss.fail()) {
            return nullptr;
        }

        auto commas = std::array<int, 5>{ 0, 0, 0, 0, 0 };
        auto cidx = 0;
        auto p = 29;

        while (cidx < 5 && p < row.length()) {
            const auto c = row.at(p);
            if (c == '\n') {
                break;
            }
            else if (c == ',') {
                commas[cidx] = p + 1;
                ++cidx;
            }
            ++p;
        }

        if (commas[0] != 30 || cidx != 5) {
            return nullptr;
        }

        const std::string symbol = row.substr(commas[0], (size_t)(commas[1] - commas[0] - 1));

        auto buy = false;
        if (row.substr(commas[1], (size_t) (commas[2] - commas[1] - 1)) == "Buy") {
            buy = true;
        }

        auto volume = float{};
        try {
            const std::string token = row.substr(commas[2], (size_t)(commas[3] - commas[2] - 1));
            volume = std::stof(token);
        }
        catch (...) {
            return nullptr; // Invalid volume format
        }

        auto price = float{};
        try {
            const std::string token = row.substr(commas[3], (size_t)(commas[4] - commas[3] - 1));
            price = std::stof(token);
        }
        catch (...) {
            return nullptr; // Invalid price format
        }

        if ((*tick_data)[symbol] == nullptr) {
            (*tick_data)[symbol] = std::make_unique<DatabaseTicks>();
        }
        (*tick_data)[symbol]->rows.push_back({ timestamp, price, volume, buy });
    }

    logger.info("BitmexDaily::parse_raw end (%d ms)", timer.elapsed().count() / 1000);

    return tick_data;

    /*
    const std::regex fieldsregx(",");
    const std::regex linesregx("\\n");

    std::string indata = raw_data.str();
    std::sregex_token_iterator row_it(indata.begin(), indata.end(), linesregx, -1);
    std::sregex_token_iterator row_end;

    ++row_it; // Skip table headers
    while (row_it != row_end) {
        const std::string row = row_it->str();
        ++row_it;
        
        std::sregex_token_iterator col_it(row.begin(), row.end(), fieldsregx, -1);
        std::sregex_token_iterator col_end;

        time_point_us timestamp;
        std::string symbol;
        float price;
        float volume;
        bool buy;
        bool valid = false;

        timer.start();
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
    */
}
