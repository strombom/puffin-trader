#include "pch.h"

#include "BitLib/BitBotConstants.h"
#include "BitLib/DateTime.h"
#include "BitLib/Logger.h"
#include "CoinbaseProTick.h"

#include <msgpack.hpp>

#include <array>
#include <regex>
#include <string>
#include <iostream>


CoinbaseProTick::CoinbaseProTick(sptrDatabase database, tick_data_updated_callback_t tick_data_updated_callback) :
    database(database), tick_data_updated_callback(tick_data_updated_callback),
    state(CoinbaseProTickState::idle), tick_data_thread_running(true)
{
    rest_api = std::make_unique<CoinbaseProRestApi>();

    tick_data_worker_thread = std::make_unique<std::thread>(&CoinbaseProTick::tick_data_worker, this);
}

CoinbaseProTickState CoinbaseProTick::get_state(void)
{
    return state;
}

void CoinbaseProTick::shutdown(void)
{
    logger.info("CoinbaseProTick::shutdown");

    state = CoinbaseProTickState::idle;

    tick_data_thread_running = false;
    tick_data_condition.notify_all();

    try {
        tick_data_worker_thread->join();
    }
    catch (...) {}
}

void CoinbaseProTick::start(void)
{
    assert(state == CoinbaseProTickState::idle);
    state = CoinbaseProTickState::downloading;
    tick_data_condition.notify_one();
}

void CoinbaseProTick::insert_symbol_name(const std::string& symbol_name)
{
    auto symbol_names = database->get_attribute(BitBase::CoinbasePro::exchange_name, "symbols", std::unordered_set<std::string>{});
    if (symbol_names.count(symbol_name) == 0) {
        symbol_names.insert(symbol_name);
        database->set_attribute(BitBase::CoinbasePro::exchange_name, "symbols", symbol_names);
    }
}

void CoinbaseProTick::tick_data_worker(void)
{
    while (tick_data_thread_running) {
        {
            auto tick_data_lock = std::unique_lock<std::mutex>{ tick_data_mutex };
            tick_data_condition.wait(tick_data_lock);
        }

        auto fetch_more = true;
        while (tick_data_thread_running && fetch_more) {
            fetch_more = false;

            for (auto&& symbol : BitBase::CoinbasePro::symbols) {
                const auto last_timestamp = database->get_attribute(BitBase::CoinbasePro::exchange_name, symbol, "tick_data_last_timestamp", BitBase::CoinbasePro::first_timestamp - 1ms);
                if (last_timestamp > system_clock_ms_now() - BitBase::CoinbasePro::Live::buffer_length) {
                    continue;
                }

                const auto last_trade_id = database->get_attribute(BitBase::CoinbasePro::exchange_name, symbol, "tick_data_last_id", (long long)BitBase::CoinbasePro::Tick::first_id - 1ll);

                auto [ticks, new_last_trade_id] = rest_api->get_aggregate_trades(symbol, last_trade_id);

                if (ticks->rows.size() == 0 || last_trade_id == new_last_trade_id) {
                    continue;
                }

                if (ticks->rows.size() > 0) {
                    logger.info("CoinbaseProTick::tick_data_worker append count(%d) (%s) (%0.1f)", (int)ticks->rows.size(), DateTime::to_string(last_timestamp).c_str(), ticks->rows.back().price);

                    if (ticks->rows.size() >= BitBase::CoinbasePro::Tick::max_rows - 1) {
                        fetch_more = true;
                    }

                    database->extend_tick_data(BitBase::CoinbasePro::exchange_name, symbol, ticks, BitBase::CoinbasePro::first_timestamp - 1ms);
                    database->set_attribute(BitBase::CoinbasePro::exchange_name, symbol, "tick_data_last_id", new_last_trade_id);
                    insert_symbol_name(symbol);
                    
                    if (fetch_more) {
                        std::this_thread::sleep_for(BitBase::CoinbasePro::Tick::rate_limit);
                    }
                }
            }
        }

        tick_data_updated_callback();
        state = CoinbaseProTickState::idle;
    }
    logger.info("CoinbaseProTick::tick_data_worker exit");
}
