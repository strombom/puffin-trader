#include "pch.h"

#include "BitLib/BitBotConstants.h"
#include "BitLib/DateTime.h"
#include "BitLib/Logger.h"
#include "BinanceLive.h"

#include <msgpack.hpp>

#include <array>
#include <regex>
#include <string>
#include <iostream>


struct BinanceMessageTick
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

    MSGPACK_DEFINE(timestamp_ms, price, volume, buy, trade_id);
};


BinanceLive::BinanceLive(sptrDatabase database, tick_data_updated_callback_t tick_data_updated_callback) :
    database(database), tick_data_updated_callback(tick_data_updated_callback),
    state(BinanceLiveState::idle), tick_data_thread_running(true)
{
    zmq_client = std::make_unique<zmq::socket_t>(zmq_context, zmq::socket_type::req);
    zmq_client->setsockopt(ZMQ_RCVTIMEO, 2500);
    zmq_client->setsockopt(ZMQ_SNDTIMEO, 2500);
    zmq_client->connect(BitBase::Binance::Live::address);

    tick_data_worker_thread = std::make_unique<std::thread>(&BinanceLive::tick_data_worker, this);
}

BinanceLiveState BinanceLive::get_state(void)
{
    return state;
}

void BinanceLive::shutdown(void)
{
    logger.info("BinanceLive::shutdown");

    state = BinanceLiveState::idle;

    tick_data_thread_running = false;
    tick_data_condition.notify_all();

    try {
        tick_data_worker_thread->join();
    }
    catch (...) {}
}

void BinanceLive::start(void)
{
    assert(state == BinanceLiveState::idle);
    state = BinanceLiveState::downloading;
    tick_data_condition.notify_one();
}

void BinanceLive::tick_data_worker(void)
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

                auto timestamp_next = database->get_attribute(BitBase::Binance::exchange_name, symbol, "tick_data_last_timestamp", BitBase::Binance::first_timestamp);
                timestamp_next += std::chrono::milliseconds{ 1 }; // Do not include the latest timestamp, only newer should be fetched

                json11::Json command = json11::Json::object{
                    { "command", "get_ticks" },
                    { "symbol", symbol },
                    { "max_rows", BitBase::Binance::Live::max_rows },
                    { "timestamp_start", DateTime::to_string_iso_8601(timestamp_next) }
                };

                auto message = zmq::message_t{ command.dump() };
                auto result = zmq::send_result_t{};
                try {
                    result = zmq_client->send(message, zmq::send_flags::dontwait);
                }
                catch (...) {
                    result.reset();
                }

                if (result.has_value()) {
                    auto result = zmq_client->recv(message);

                    if (result.has_value()) {
                        auto received_ticks = std::vector<BinanceMessageTick>{};
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
                        auto last_id = 0ll;
                        for (auto& tick : received_ticks) {
                            if (tick.timestamp() < last_timestamp) {
                                tick_data->rows.push_back({ tick.timestamp(), tick.price, tick.volume, tick.buy });
                                last_id = tick.trade_id;
                            }
                        }

                        if (tick_data->rows.size() > 0) {
                            logger.info("BinanceLive::tick_data_worker append count(%d) (%s) (%0.1f)", (int)tick_data->rows.size(), DateTime::to_string(last_timestamp).c_str(), tick_data->rows.back().price);

                            database->extend_tick_data(BitBase::Binance::exchange_name, symbol, std::move(tick_data), BitBase::Binance::first_timestamp);
                            database->set_attribute(BitBase::Binance::exchange_name, symbol, "tick_data_last_id", last_id);
                            if (received_ticks.size() >= BitBase::Binance::Live::max_rows - 1) {
                                fetch_more = true;
                            }
                        }
                    }
                    else {
                        logger.info("BinanceLive::tick_data_worker zmq recv fail");
                    }
                }
                else {
                    logger.info("BinanceLive::tick_data_worker zmq send fail");
                }
            }
        }

        tick_data_updated_callback();
        state = BinanceLiveState::idle;
    }
    logger.info("BinanceDaily::tick_data_worker exit");
}
