#include "pch.h"

#include "BitLib/BitBotConstants.h"
#include "BitLib/DateTime.h"
#include "BitLib/Logger.h"
#include "BinanceTick.h"

#include <msgpack.hpp>

#include <array>
#include <regex>
#include <string>
#include <iostream>


BinanceTick::BinanceTick(sptrDatabase database, tick_data_updated_callback_t tick_data_updated_callback) :
    database(database), tick_data_updated_callback(tick_data_updated_callback),
    state(BinanceTickState::idle), tick_data_thread_running(true)
{
    rest_api = std::make_unique<BinanceRestApi>();

    tick_data_worker_thread = std::make_unique<std::thread>(&BinanceTick::tick_data_worker, this);
}

BinanceTickState BinanceTick::get_state(void)
{
    return state;
}

void BinanceTick::shutdown(void)
{
    logger.info("BinanceTick::shutdown");

    state = BinanceTickState::idle;

    tick_data_thread_running = false;
    tick_data_condition.notify_all();

    try {
        tick_data_worker_thread->join();
    }
    catch (...) {}
}

void BinanceTick::start(void)
{
    assert(state == BinanceTickState::idle);
    state = BinanceTickState::downloading;
    tick_data_condition.notify_one();
}

void BinanceTick::insert_symbol_name(const std::string& symbol_name)
{
    auto symbol_names = database->get_attribute(BitBase::Binance::exchange_name, "symbols", std::unordered_set<std::string>{});
    if (symbol_names.count(symbol_name) == 0) {
        symbol_names.insert(symbol_name);
        database->set_attribute(BitBase::Binance::exchange_name, "symbols", symbol_names);
    }
}

void BinanceTick::tick_data_worker(void)
{
    while (tick_data_thread_running) {
        {
            auto tick_data_lock = std::unique_lock<std::mutex>{ tick_data_mutex };
            tick_data_condition.wait(tick_data_lock);
        }

        auto fetch_more = true;
        while (tick_data_thread_running && fetch_more) {
            fetch_more = false;

            for (auto&& symbol : BitBase::Binance::symbols) {
                auto last_id = database->get_attribute(BitBase::Binance::exchange_name, symbol, "tick_data_last_id", -1ll);
                auto timestamp_next = database->get_attribute(BitBase::Binance::exchange_name, symbol, "tick_data_last_timestamp", BitBase::Binance::first_timestamp - 1ms);
                timestamp_next += 1ms; // Do not include the latest timestamp, only newer should be fetched

                if (last_id == -1) {
                    timestamp_next -= 1h;
                }

                auto [ticks, new_last_id] = rest_api->get_aggregate_trades(symbol, last_id, timestamp_next);

                if (ticks->rows.size() == 0) {
                    continue;
                }

                const auto last_timestamp = ticks->rows.back().timestamp;

                if (ticks->rows.size() > 0) {
                    logger.info("BinanceLive::tick_data_worker append count(%d) (%s) (%0.1f)", (int)ticks->rows.size(), DateTime::to_string(last_timestamp).c_str(), ticks->rows.back().price);

                    if (ticks->rows.size() >= BitBase::Binance::Tick::max_rows - 1) {
                        fetch_more = true;
                    }
                    // Potential bug, might skip multiple ticks on the same timestamp, unlikely to occur so we don't mind - 2020-06-22
                    database->extend_tick_data(BitBase::Binance::exchange_name, symbol, std::move(ticks), BitBase::Binance::first_timestamp - 1ms);
                    database->set_attribute(BitBase::Binance::exchange_name, symbol, "tick_data_last_id", new_last_id);
                    insert_symbol_name(symbol);
                }
            }
        }

        tick_data_updated_callback();
        state = BinanceTickState::idle;
    }
    logger.info("BinanceTick::tick_data_worker exit");
}
