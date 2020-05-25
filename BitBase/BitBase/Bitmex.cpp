#include "pch.h"

#include "Bitmex.h"
#include "Logger.h"
#include "DateTime.h"


Bitmex::Bitmex(sptrDatabase database, sptrDownloadManager download_manager) :
    database(database), download_manager(download_manager), state(BitmexState::idle), thread_running(true)
{
    bitmex_daily = std::make_unique<BitmexDaily>(database, download_manager, std::bind(&Bitmex::tick_data_updated_callback, this));
    bitmex_interim = std::make_unique<BitmexInterim>(database, std::bind(&Bitmex::tick_data_updated_callback, this));
    bitmex_live = std::make_unique<BitmexLive>(database, std::bind(&Bitmex::tick_data_updated_callback, this));
    bitmex_interval = std::make_unique<BitmexInterval>(database);

    main_loop_thread = std::make_unique<std::thread>(&Bitmex::main_loop, this);
    interval_update_thread = std::make_unique<std::thread>(&Bitmex::interval_update_worker, this);
}

void Bitmex::shutdown(void)
{
    logger.info("Bitmex::shutdown");
    {
        auto slock = std::scoped_lock{ state_mutex };
        state = BitmexState::shutdown;
    }
    logger.info("Bitmex::shutdown state = shutdown");
    bitmex_daily->shutdown();
    bitmex_interval->shutdown();

    try {
        main_loop_thread->join();
    }
    catch (...) {}

    try {
        interval_update_condition.notify_all();
        interval_update_thread->join();
    }
    catch (...) {}
}

void Bitmex::tick_data_updated_callback(void)
{
    interval_update_condition.notify_one();
}

/*
#include "BitmexAPI/ApiClient.h"
#include "BitmexAPI/api/TradeApi.h"
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

auto conf = bitmex_api->getConfiguration();

const auto bu = conf->getBaseUrl();

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
void Bitmex::main_loop(void)
{
    while (state != BitmexState::shutdown) {
        {
            auto slock = std::scoped_lock{ state_mutex };

            if (state == BitmexState::idle) {
                auto tick_data_last_timestamp = database->get_attribute("BITMEX", "XBTUSD", "tick_data_last_timestamp", BitBase::Bitmex::first_timestamp);
                if (tick_data_last_timestamp < std::chrono::system_clock::now() - std::chrono::hours{ 24 + 6 }) {
                    // Last tick timestamp is more than 24 + 6 hours old, there should be a compressed daily archive available
                    state = BitmexState::downloading_daily;
                    bitmex_daily->start();
                }
                else if (tick_data_last_timestamp < std::chrono::system_clock::now() - std::chrono::minutes{ 1 }) {
                    state = BitmexState::downloading_interim;
                    bitmex_interim->start();
                }
                else {
                    state = BitmexState::downloading_live;
                    bitmex_live->start();
                }

            } else if (state == BitmexState::downloading_daily) {
                // Check if daily data is downloaded
                if (bitmex_daily->get_state() == BitmexDailyState::idle) {
                    state = BitmexState::idle;
                }
            }
            else if (state == BitmexState::downloading_interim) {
                // Check if interim data is up to date
                if (bitmex_interim->get_state() == BitmexInterimState::idle) {
                    state = BitmexState::idle;
                }
            }
            else if (state == BitmexState::downloading_live) {
                // Check if live data has stopped
                if (bitmex_live->get_state() == BitmexLiveState::idle) {
                    state = BitmexState::idle;
                }
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
}

void Bitmex::interval_update_worker(void)
{
    while (state != BitmexState::shutdown) {
        {
            auto interval_update_lock = std::unique_lock<std::mutex>{ interval_update_mutex };
            interval_update_condition.wait(interval_update_lock);
        }

        bitmex_interval->update();
    }
}
