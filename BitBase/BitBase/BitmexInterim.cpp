#include "pch.h"

#include "BitBotConstants.h"
#include "BitmexInterim.h"
#include "DateTime.h"
#include "Logger.h"

#include <regex>
#include <string>
#include <array>

/*
logger.info("start");

auto bitmex_config = std::make_shared<io::swagger::client::api::ApiConfiguration>(io::swagger::client::api::ApiConfiguration{});

auto api_key_id = utility::string_t{ L"ynOrYOWoC1knanjDld9RtPhC" };
auto api_key_secret = utility::string_t{ L"0d_jDIPan7mEHSPhQDyMQJKVPJ3kEc5qbS5ed5JBWiKIsAXW" };
auto base_url = utility::string_t{ L"https://www.bitmex.com/api/v1" };
auto http_config = web::http::client::http_client_config{};
auto user_agent = utility::string_t{ L"abc" };

bitmex_config->setApiKey(api_key_id, api_key_secret);
bitmex_config->setBaseUrl(base_url);
bitmex_config->setHttpConfig(http_config);
bitmex_config->setUserAgent(user_agent);

auto bitmex_api = std::make_shared< io::swagger::client::api::ApiClient>(bitmex_config);

auto trade_api = io::swagger::client::api::TradeApi{ bitmex_api };

const auto symbol = utility::string_t{ L"XBTUSD" };
const auto count = 1000;
const auto start_time = utility::datetime::from_string(L"2020-05-24T00:00:00Z", utility::datetime::date_format::ISO_8601);
const auto end_time = utility::datetime::from_string(L"2020-05-25T00:00:00Z", utility::datetime::date_format::ISO_8601);
auto trade = trade_api.trade_get(
    symbol,
    boost::none,
    boost::none,
    count,
    boost::none,
    boost::none,
    start_time,
    end_time
);

auto results = trade.get();

for (auto result : results) {
    std::wcout <<
        "Result: " << result->getTimestamp().to_string(utility::datetime::date_format::ISO_8601).c_str() <<
        "  " << result->getPrice() <<
        "  " << result->getSize() <<
        std::endl;
    break;
}

std::cout << "Count: " << results.size() << std::endl;

return 1;
*/

#include <chrono>

BitmexInterim::BitmexInterim(sptrDatabase database, tick_data_updated_callback_t tick_data_updated_callback) :
    database(database), tick_data_updated_callback(tick_data_updated_callback),
    state(BitmexInterimState::idle), tick_data_thread_running(true)
{
    // Init Bitmex API
    const auto bitmex_config = std::make_shared<io::swagger::client::api::ApiConfiguration>(io::swagger::client::api::ApiConfiguration{});
    auto http_config = web::http::client::http_client_config{};
    bitmex_config->setHttpConfig(http_config);
    bitmex_config->setApiKey(BitBase::Bitmex::Interim::api_key_id, BitBase::Bitmex::Interim::api_key_secret);
    bitmex_config->setBaseUrl(BitBase::Bitmex::Interim::base_url);
    bitmex_config->setUserAgent(L"unkwnown");
    auto bitmex_api = std::make_shared<io::swagger::client::api::ApiClient>(bitmex_config);
    trade_api = std::make_unique<io::swagger::client::api::TradeApi>(bitmex_api);

    timestamp_next = database->get_attribute(BitBase::Bitmex::exchange_name, BitBase::Bitmex::Interim::symbol, "tick_data_last_timestamp", time_point_ms{ BitBase::Bitmex::first_timestamp });

    tick_data_worker_thread = std::make_unique<std::thread>(&BitmexInterim::tick_data_worker, this);
}

BitmexInterimState BitmexInterim::get_state(void)
{
    return state;
}

void BitmexInterim::shutdown(void)
{
    logger.info("BitmexDaily::shutdown");

    {
        // Will not start new downloads after this section
        auto slock = std::scoped_lock{ start_download_mutex };
        state = BitmexInterimState::idle;
    }

    tick_data_thread_running = false;
    tick_data_condition.notify_all();

    //download_manager->abort_client(BitBase::Bitmex::Daily::downloader_client_id);

    try {
        tick_data_worker_thread->join();
    }
    catch (...) {}
}

void BitmexInterim::start(void)
{
    assert(state == BitmexInterimState::idle);
    
    state = BitmexInterimState::downloading;

    if (timestamp_next > date::floor<date::days>(system_clock_us_now() - std::chrono::minutes{ 1 })) {
        state = BitmexInterimState::idle;
        return;
    }


    const auto symbol = utility::string_t{ BitBase::Bitmex::Interim::symbol_w };
    const auto count = 50;
    const auto start_time = utility::datetime::from_string(DateTime::to_string_iso_8601(timestamp_next), utility::datetime::date_format::ISO_8601);

    auto trade = trade_api->trade_get(
        symbol,
        boost::none,
        boost::none,
        count,
        boost::none,
        boost::none,
        start_time,
        boost::none
    );

    auto results = trade.get();

    for (auto result : results) {
        std::wcout <<
            "Result: " << result->getTimestamp().to_string(utility::datetime::date_format::ISO_8601).c_str() <<
            "  " << result->getPrice() <<
            "  " << result->getSize() <<
            std::endl;
    }

    std::cout << "Count: " << results.size() << std::endl;

    std::wcout << "First w: " << DateTime::to_string_iso_8601(timestamp_next) << std::endl;
    std::cout << "First s: " << DateTime::to_string(timestamp_next) << std::endl;



    auto url = std::string{ BitBase::Bitmex::Daily::base_url_start } + date::format(BitBase::Bitmex::Daily::url_date_format, timestamp_next) + std::string{ BitBase::Bitmex::Daily::base_url_end };
    //download_manager->download(url, BitBase::Bitmex::Daily::downloader_client_id, std::bind(&BitmexDaily::download_done_callback, this, std::placeholders::_1));
    timestamp_next += date::days{ 1 };
}

/*
void BitmexInterim::download_done_callback(sptr_download_data_t payload)
{
    auto compressed = boost::iostreams::array_source{ payload->data(), payload->size() };
    auto out = boost::iostreams::filtering_streambuf<boost::iostreams::input>{};
    out.push(boost::iostreams::gzip_decompressor{});
    out.push(compressed);

    auto decompressed = std::stringstream{};
    boost::iostreams::copy(out, decompressed);
    auto tick_data = parse_raw(decompressed);
    //(*tick_data)[symbol]->rows.push_back({ timestamp, price, volume, buy });

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
}
*/

void BitmexInterim::update_symbol_names(const std::unordered_set<std::string>& new_symbol_names)
{
    auto symbol_names = database->get_attribute(BitBase::Bitmex::exchange_name, "symbols", std::unordered_set<std::string>{});
    for (auto&& symbol_name : new_symbol_names) {
        symbol_names.insert(symbol_name);
    }
    database->set_attribute(BitBase::Bitmex::exchange_name, "symbols", symbol_names);
}

void BitmexInterim::tick_data_worker(void)
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
