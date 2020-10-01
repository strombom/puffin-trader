#include "pch.h"

#include "BitLib/Logger.h"
#include "LiveData.h"

#include <msgpack.hpp>


struct BitmexMessageTick
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

LiveData::LiveData(time_point_ms start_time) :
    tick_data_thread_running(true),
    next_timestamp(start_time),
    agg_tick_idx(0)
{
    live_data_connect();

    tick_data_worker_thread = std::make_unique<std::thread>(&LiveData::tick_data_worker, this);
}

void LiveData::live_data_connect(void)
{
    zmq_client = std::make_unique<zmq::socket_t>(zmq_context, zmq::socket_type::req);
    zmq_client->setsockopt(ZMQ_RCVTIMEO, 2500);
    zmq_client->setsockopt(ZMQ_SNDTIMEO, 2500);
    zmq_client->connect(BitBase::Bitmex::Live::address);
}

void LiveData::start(void)
{
    //live_data_thread = std::make_unique<std::thread>(&LiveData::live_data_worker, this);

    tick_data_condition.notify_one();
}

void LiveData::shutdown(void)
{
    std::cout << "LiveData: Shutting down" << std::endl;
    //live_data_thread_running = false;
    tick_data_thread_running = false;
    tick_data_condition.notify_all();

    //try {
    //    live_data_thread->join();
    //}
    //catch (...) {}

    try {
        tick_data_worker_thread->join();
    }
    catch (...) {}
}

sptrAggTick LiveData::get_next_agg_tick(void)
{
    auto lock = std::scoped_lock{ agg_ticks_mutex };
    if (agg_ticks.agg_ticks.size() > agg_tick_idx) {
        // TODO: If agg_tick_idx, remove elements from beginning of agg_ticks
        return std::make_shared<AggTick>(agg_ticks.agg_ticks[agg_tick_idx++]);
    }
    return nullptr;
}


void LiveData::tick_data_worker(void)
{
    while (tick_data_thread_running) {
        {
            auto tick_data_lock = std::unique_lock<std::mutex>{ tick_data_mutex };
            tick_data_condition.wait(tick_data_lock);
        }

        while (tick_data_thread_running) {
            for (auto&& symbol : BitBase::Bitmex::symbols) {

                next_timestamp += 1ms; // Do not include the latest timestamp, only newer should be fetched

                json11::Json command = json11::Json::object{
                    { "command", "get_ticks" },
                    { "symbol", symbol },
                    { "max_rows", BitBase::Bitmex::Live::max_rows },
                    { "timestamp_start", DateTime::to_string_iso_8601(next_timestamp) }
                };

                auto message = zmq::message_t{ command.dump() };
                auto result = zmq::send_result_t{};
                try {
                    result = zmq_client->send(message, zmq::send_flags::dontwait);
                }
                catch (...) {
                    result.reset();
                    live_data_connect();
                }

                if (result.has_value()) {
                    auto result = zmq_client->recv(message);

                    if (result.has_value()) {
                        auto received_ticks = std::vector<BitmexMessageTick>{};
                        received_ticks = msgpack::unpack(static_cast<const char*>(message.data()), message.size()).get().convert(received_ticks);

                        auto last_timepoint = std::chrono::system_clock::time_point(std::chrono::system_clock::now() - 500ms);
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

                                auto lock = std::scoped_lock{ agg_ticks_mutex };
                                agg_ticks.insert(Tick{ tick.timestamp(), tick.price, tick.volume, tick.buy });
                                next_timestamp = tick.timestamp();

                            }
                        }

                        if (tick_data->rows.size() > 0) {
                            //logger.info("BitmexLive::tick_data_worker append count(%d) (%s) (%0.1f)", (int)tick_data->rows.size(), DateTime::to_string(last_timestamp).c_str(), tick_data->rows.back().price);
                        }
                    }
                    else {
                        logger.info("BitmexLive::tick_data_worker zmq recv fail");
                    }
                }
                else {
                    logger.info("BitmexLive::tick_data_worker zmq send fail");
                }

                std::this_thread::sleep_for(10ms);
            }
        }
    }
    logger.info("BitmexDaily::tick_data_worker exit");
}
