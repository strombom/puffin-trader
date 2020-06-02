#include "pch.h"

#include "BitBotConstants.h"
#include "BitmexLive.h"
#include "DateTime.h"
#include "Logger.h"

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


BitmexLive::BitmexLive(sptrDatabase database, tick_data_updated_callback_t tick_data_updated_callback) :
    database(database), tick_data_updated_callback(tick_data_updated_callback),
    state(BitmexLiveState::idle), tick_data_thread_running(true)
{
    zmq_client = std::make_unique<zmq::socket_t>(zmq_context, zmq::socket_type::req);
    zmq_client->connect(BitBase::Bitmex::Live::address);

    tick_data_worker_thread = std::make_unique<std::thread>(&BitmexLive::tick_data_worker, this);
}

BitmexLiveState BitmexLive::get_state(void)
{
    return state;
}

void BitmexLive::shutdown(void)
{
    logger.info("BitmexLive::shutdown");
    
    state = BitmexLiveState::idle;

    tick_data_thread_running = false;
    tick_data_condition.notify_all();

    try {
        tick_data_worker_thread->join();
    }
    catch (...) {}
}

void BitmexLive::start(void)
{
    std::cout << "BitmexLive::start" << std::endl;
    assert(state == BitmexLiveState::idle);
    
    state = BitmexLiveState::downloading;

    tick_data_condition.notify_one();
}

void BitmexLive::tick_data_worker(void)
{
    while (tick_data_thread_running) {
        {
            auto tick_data_lock = std::unique_lock<std::mutex>{ tick_data_mutex };
            tick_data_condition.wait(tick_data_lock);
        }

        auto has_data = true;
        while (tick_data_thread_running && has_data) {
            has_data = false;

            for (auto&& symbol : BitBase::Bitmex::symbols) {

                std::cout << "Tick data " << symbol << std::endl;

                auto timestamp_next = database->get_attribute(BitBase::Bitmex::exchange_name, symbol, "tick_data_last_timestamp", BitBase::Bitmex::first_timestamp);

                std::cout << "tscount " << timestamp_next.time_since_epoch().count() << std::endl;
                std::cout << "ftcount " << BitBase::Bitmex::first_timestamp.time_since_epoch().count() << std::endl;

                auto timestamp_min = DateTime::to_time_point_ms("2020-06-02T15:50:00.000Z", "%FT%TZ");

                auto timestamp_string = std::string{};

                if (timestamp_next < timestamp_min) {
                    timestamp_string = DateTime::to_string_iso_8601(timestamp_min);
                }
                else {
                    timestamp_string = DateTime::to_string_iso_8601(timestamp_next);
                }

                //const auto temp_timestamp = std::chrono::system_clock::now() - std::chrono::seconds{ 60 };
                //const auto temp_ms = std::chrono::milliseconds(temp_timestamp.time_since_epoch().count());
                //const auto temp_time_point = time_point_ms{ temp_ms };

                //auto tp = std::chrono::system_clock::time_point(std::chrono::system_clock::now() - std::chrono::seconds{ 4 });
                //auto temp_time_point = std::chrono::time_point_cast<std::chrono::time_point<std::chrono::system_clock, std::chrono::milliseconds>::duration>( tp );
                //const auto dstr = DateTime::to_string_iso_8601(temp_time_point);

                json11::Json command = json11::Json::object{
                    { "command", "get_ticks" },
                    { "symbol", symbol },
                    { "max_rows", BitBase::Bitmex::Live::max_rows },
                    { "timestamp_start", timestamp_string } //DateTime::to_string_iso_8601(timestamp_next) }
                };

                std::cout << "Send command: " << command.dump() << std::endl;

                auto message = zmq::message_t{ command.dump() };
                auto result = zmq_client->send(message, zmq::send_flags::dontwait);

                std::cout << "Send response: " << result.has_value() << std::endl;
                std::cout << "Send response: " << result.value() << std::endl;
                

                if (result.has_value()) {
                    auto result = zmq_client->recv(message);

                    auto received_ticks = std::vector<MessageTick>{};
                    received_ticks = msgpack::unpack(static_cast<const char*>(message.data()), message.size()).get().convert(received_ticks);

                    // DONT APPEND THE LAST TIMESTAMP!

                    auto tick_data = std::make_unique<Ticks>();
                    for (auto &tick: received_ticks) {
                        std::cout << "Recv: " << DateTime::to_string(tick.timestamp())
                            << " " << tick.price
                            << " " << tick.volume
                            << " " << tick.buy
                            << std::endl;
                        tick_data->rows.push_back({ tick.timestamp(), tick.price, tick.volume, tick.buy });
                    }


                    database->extend_tick_data(BitBase::Bitmex::exchange_name, symbol, std::move(tick_data), BitBase::Bitmex::first_timestamp);

                    logger.info("BitmexLive::tick_data_worker tick_data appended to database count(%d)", received_ticks.size());

                    //update_symbol_names(symbol_names);

                    has_data = true;

                    std::cout << "Recv count: " << message.size() << std::endl;
                }
                else {
                    std::cout << "Recv: fail" << std::endl;
                }
            }

            state = BitmexLiveState::idle;
            break;
        }
        
        tick_data_updated_callback();
    }
    logger.info("BitmexDaily::tick_data_worker exit");
}
