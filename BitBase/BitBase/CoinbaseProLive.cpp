#include "pch.h"

#include "BitLib/BitBotConstants.h"
#include "BitLib/DateTime.h"
#include "BitLib/Logger.h"
#include "CoinbaseProLive.h"

#include <msgpack.hpp>

#include <array>
#include <regex>
#include <string>
#include <iostream>


struct CoinbaseProMessageTick
{
public:
    unsigned long long timestamp_ms;
    float price;
    float volume;
    bool buy;
    long long trade_id;

    time_point_ms timestamp(void)
    {
        return std::chrono::time_point<std::chrono::system_clock, std::chrono::milliseconds>{ std::chrono::milliseconds{ timestamp_ms } };
    }

    MSGPACK_DEFINE(timestamp_ms, price, volume, buy);
};


CoinbaseProLive::CoinbaseProLive(sptrDatabase database, tick_data_updated_callback_t tick_data_updated_callback) :
    database(database), tick_data_updated_callback(tick_data_updated_callback),
    state(CoinbaseProLiveState::idle), tick_data_thread_running(true)
{
    zmq_client = std::make_unique<zmq::socket_t>(zmq_context, zmq::socket_type::req);
    zmq_client->connect(BitBase::CoinbasePro::Live::address);

    tick_data_worker_thread = std::make_unique<std::thread>(&CoinbaseProLive::tick_data_worker, this);
}

CoinbaseProLiveState CoinbaseProLive::get_state(void)
{
    return state;
}

void CoinbaseProLive::shutdown(void)
{
    logger.info("CoinbaseProLive::shutdown");

    state = CoinbaseProLiveState::idle;

    tick_data_thread_running = false;
    tick_data_condition.notify_all();

    try {
        tick_data_worker_thread->join();
    }
    catch (...) {}
}

void CoinbaseProLive::start(void)
{
    assert(state == CoinbaseProLiveState::idle);
    state = CoinbaseProLiveState::downloading;
    tick_data_condition.notify_one();
}

void CoinbaseProLive::tick_data_worker(void)
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

                auto timestamp_next = database->get_attribute(BitBase::CoinbasePro::exchange_name, symbol, "tick_data_last_timestamp", BitBase::CoinbasePro::first_timestamp);
                timestamp_next += std::chrono::milliseconds{ 1 }; // Do not include the latest timestamp, only newer should be fetched

                json11::Json command = json11::Json::object{
                    { "command", "get_ticks" },
                    { "symbol", symbol },
                    { "max_rows", BitBase::CoinbasePro::Live::max_rows },
                    { "timestamp_start", DateTime::to_string_iso_8601(timestamp_next) }
                };

                auto message = zmq::message_t{ command.dump() };
                auto result = zmq_client->send(message, zmq::send_flags::dontwait);

                if (result.has_value()) {
                    auto result = zmq_client->recv(message);

                    auto received_ticks = std::vector<CoinbaseProMessageTick>{};
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
                    auto last_trade_id = 0ll;
                    for (auto& tick : received_ticks) {
                        if (tick.timestamp() < last_timestamp) {
                            tick_data->rows.push_back({ tick.timestamp(), tick.price, tick.volume, tick.buy });
                            last_trade_id = tick.trade_id;
                        }
                    }

                    if (tick_data->rows.size() > 0) {
                        logger.info("CoinbaseProLive::tick_data_worker append count(%d) (%s) (%0.1f)", (int)tick_data->rows.size(), DateTime::to_string(last_timestamp).c_str(), tick_data->rows.back().price);

                        database->extend_tick_data(BitBase::CoinbasePro::exchange_name, symbol, std::move(tick_data), BitBase::CoinbasePro::first_timestamp);
                        database->set_attribute(BitBase::CoinbasePro::exchange_name, symbol, "tick_data_last_id", last_trade_id);
                        if (received_ticks.size() >= BitBase::CoinbasePro::Live::max_rows - 1) {
                            fetch_more = true;
                        }
                    }
                }
                else {
                    logger.info("CoinbaseProLive::tick_data_worker fail");
                }
            }
        }

        tick_data_updated_callback();
        state = CoinbaseProLiveState::idle;
    }
    logger.info("CoinbaseProDaily::tick_data_worker exit");
}
