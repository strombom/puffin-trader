#include "pch.h"

#include "BitBotConstants.h"
#include "BitmexInterim.h"
#include "DateTime.h"
#include "Logger.h"

#include <array>
#include <regex>
#include <chrono>
#include <string>


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

    for (auto symbol : BitBase::Bitmex::symbols) {
        timestamps_next[symbol] = database->get_attribute(BitBase::Bitmex::exchange_name, symbol, "tick_data_last_timestamp", time_point_ms{ BitBase::Bitmex::first_timestamp });
    }

    tick_data_worker_thread = std::make_unique<std::thread>(&BitmexInterim::tick_data_worker, this);
}

BitmexInterimState BitmexInterim::get_state(void)
{
    return state;
}

void BitmexInterim::shutdown(void)
{
    logger.info("BitmexInterim::shutdown");

    {
        // Will not start new downloads after this section
        auto slock = std::scoped_lock{ start_download_mutex };
        state = BitmexInterimState::idle;
    }

    tick_data_thread_running = false;
    tick_data_condition.notify_all();

    try {
        tick_data_worker_thread->join();
    }
    catch (...) {}
}

void BitmexInterim::start(void)
{
    assert(state == BitmexInterimState::idle);    
    state = BitmexInterimState::downloading;
    tick_data_condition.notify_one();
}

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

        for (auto symbol : BitBase::Bitmex::symbols) {

            while (tick_data_thread_running) {
                if (timestamps_next[symbol] > system_clock_us_now() - std::chrono::minutes{ 1 }) {
                    break;
                }

                const auto trade_symbol = utility::conversions::to_utf16string(symbol);
                const auto trade_count = 1000;
                const auto trade_start_time = utility::datetime::from_string(DateTime::to_string_iso_8601(timestamps_next[symbol]), utility::datetime::date_format::ISO_8601);
                auto trade = trade_api->trade_get(trade_symbol, boost::none, boost::none, trade_count, boost::none, boost::none, trade_start_time, boost::none);
                auto results = trade.get();

                auto tick_data = std::make_unique<Ticks>();
                for (auto result : results) {
                    const auto timestamp_string = utility::conversions::to_utf8string(result->getTimestamp().to_string(utility::datetime::date_format::ISO_8601));
                    const auto timestamp = DateTime::to_time_point_ms(timestamp_string, "%FT%TZ");
                    const auto price = (float)result->getPrice();
                    const auto volume = (float)result->getSize();
                    const auto direction_string = utility::conversions::to_utf8string(result->getTickDirection());
                    const auto direction = (direction_string.find("PlusTick") != std::string::npos) ? true : false;
                    tick_data->rows.push_back({ timestamp, price, volume, direction });

                    
                    std::cout << "Tick "
                        << DateTime::to_string(timestamp) << " "
                        << price << " "
                        << volume << " "
                        << direction << " "
                        << std::endl;
                    
                }

                std::cout << "======" << std::endl;

                auto last_unique_idx = -1;
                auto last_unique_timestamp = tick_data->rows[tick_data->rows.size() - 1].timestamp;

                for (auto idx = (int)tick_data->rows.size() - 1; idx >= 0; --idx) {
                    
                    std::cout << "Tick "
                        << DateTime::to_string(tick_data->rows[idx].timestamp) << " "
                        << tick_data->rows[idx].price << " "
                        << tick_data->rows[idx].volume << " "
                        << tick_data->rows[idx].buy << " "
                        << std::endl;
                    

                    if (tick_data->rows[idx].timestamp != last_unique_timestamp) {
                        last_unique_idx = idx;
                        break;
                    }
                }

                if (last_unique_idx < 0) {
                    // No new data, continue with next symbol
                    continue;
                }

                std::cout << "======" << std::endl;

                tick_data->rows.resize(last_unique_idx + 1);
                
                for (auto row : tick_data->rows) {
                    std::cout << "Tick "
                        << DateTime::to_string(row.timestamp) << " "
                        << row.price << " "
                        << row.volume << " "
                        << row.buy << " "
                        << std::endl;
                }
                

                std::cout << "Last unique " << last_unique_idx << " - " << DateTime::to_string(last_unique_timestamp) << std::endl;

                database->extend_tick_data(BitBase::Bitmex::exchange_name, symbol, std::move(tick_data), BitBase::Bitmex::first_timestamp);
                timestamps_next[symbol] = database->get_attribute(BitBase::Bitmex::exchange_name, symbol, "tick_data_last_timestamp", time_point_ms{ BitBase::Bitmex::first_timestamp });

            }

            tick_data_updated_callback();
            logger.info("BitmexInterim::tick_data_worker tick_data appended to database");
        }
    }
    logger.info("BitmexInterim::tick_data_worker exit");
}
