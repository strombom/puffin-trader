#include "pch.h"

#include "BitLib/BitBotConstants.h"
#include "BitLib/DateTime.h"
#include "BitLib/Logger.h"
#include "CoinbaseLive.h"

#include <msgpack.hpp>

#include <array>
#include <regex>
#include <string>
#include <iostream>


struct MessageTick
{
public:
    unsigned long long timestamp_ms;
    float price;
    float volume;
    bool buy;

    time_point_ms timestamp(void)
    {
        return std::chrono::time_point<std::chrono::system_clock, std::chrono::milliseconds>{ std::chrono::milliseconds{ timestamp_ms } };
    }

    MSGPACK_DEFINE(timestamp_ms, price, volume, buy);
};


CoinbaseLive::CoinbaseLive(sptrDatabase database, tick_data_updated_callback_t tick_data_updated_callback) :
    database(database), tick_data_updated_callback(tick_data_updated_callback),
    state(CoinbaseLiveState::idle), tick_data_thread_running(true)
{
    zmq_client = std::make_unique<zmq::socket_t>(zmq_context, zmq::socket_type::req);
    zmq_client->connect(BitBase::Coinbase::Live::address);

    tick_data_worker_thread = std::make_unique<std::thread>(&CoinbaseLive::tick_data_worker, this);
}

CoinbaseLiveState CoinbaseLive::get_state(void)
{
    return state;
}

void CoinbaseLive::shutdown(void)
{
    logger.info("CoinbaseLive::shutdown");

    state = CoinbaseLiveState::idle;

    tick_data_thread_running = false;
    tick_data_condition.notify_all();

    try {
        tick_data_worker_thread->join();
    }
    catch (...) {}
}

void CoinbaseLive::start(void)
{
    assert(state == CoinbaseLiveState::idle);
    state = CoinbaseLiveState::downloading;
    tick_data_condition.notify_one();
}

void CoinbaseLive::tick_data_worker(void)
{
    while (tick_data_thread_running) {
        {
            auto tick_data_lock = std::unique_lock<std::mutex>{ tick_data_mutex };
            tick_data_condition.wait(tick_data_lock);
        }

        auto fetch_more = true;
        while (tick_data_thread_running && fetch_more) {
            fetch_more = false;

            for (auto&& symbol : BitBase::Coinbase::symbols) {

                auto timestamp_next = database->get_attribute(BitBase::Coinbase::exchange_name, symbol, "tick_data_last_timestamp", BitBase::Coinbase::first_timestamp);
                timestamp_next += std::chrono::milliseconds{ 1 }; // Do not include the latest timestamp, only newer should be fetched

                json11::Json command = json11::Json::object{
                    { "command", "get_ticks" },
                    { "symbol", symbol },
                    { "max_rows", BitBase::Coinbase::Live::max_rows },
                    { "timestamp_start", DateTime::to_string_iso_8601(timestamp_next) }
                };

                auto message = zmq::message_t{ command.dump() };
                auto result = zmq_client->send(message, zmq::send_flags::dontwait);

                if (result.has_value()) {
                    auto result = zmq_client->recv(message);

                    auto received_ticks = std::vector<MessageTick>{};
                    received_ticks = msgpack::unpack(static_cast<const char*>(message.data()), message.size()).get().convert(received_ticks);

                    auto last_timepoint = std::chrono::system_clock::time_point(std::chrono::system_clock::now() - std::chrono::seconds{ 5 });
                    if (received_ticks.size() > 1) {
                        if (received_ticks.back().timestamp() < last_timepoint) {
                            last_timepoint = received_ticks.back().timestamp();
                        }
                        else {
                            last_timepoint = std::chrono::system_clock::time_point(std::chrono::system_clock::now());
                        }
                    }
                    auto last_timestamp = std::chrono::time_point_cast<std::chrono::time_point<std::chrono::system_clock, std::chrono::milliseconds>::duration>(last_timepoint);

                    auto tick_data = std::make_unique<Ticks>();
                    for (auto& tick : received_ticks) {
                        if (tick.timestamp() < last_timestamp) {
                            tick_data->rows.push_back({ tick.timestamp(), tick.price, tick.volume, tick.buy });
                        }
                    }

                    if (tick_data->rows.size() > 0) {
                        logger.info("CoinbaseLive::tick_data_worker append count(%d) (%s) (%0.1f)", (int)tick_data->rows.size(), DateTime::to_string(last_timestamp).c_str(), tick_data->rows.back().price);

                        database->extend_tick_data(BitBase::Coinbase::exchange_name, symbol, std::move(tick_data), BitBase::Coinbase::first_timestamp);
                        if (received_ticks.size() >= BitBase::Coinbase::Live::max_rows - 1) {
                            fetch_more = true;
                        }
                    }
                }
                else {
                    logger.info("CoinbaseLive::tick_data_worker fail");
                }
            }
        }

        tick_data_updated_callback();
        state = CoinbaseLiveState::idle;
    }
    logger.info("CoinbaseDaily::tick_data_worker exit");
}
