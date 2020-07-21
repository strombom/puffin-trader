#include "pch.h"

#include "Logger.h"
#include "BitBaseClient.h"
#include "BitBotConstants.h"


BitBaseClient::BitBaseClient(void)
{
    client = std::make_unique<zmq::socket_t>(context, zmq::socket_type::req);
    client->connect(BitSim::BitBase::address);
}

sptrIntervals BitBaseClient::get_intervals(const std::string& symbol,
                                           const std::string& exchange,
                                           const time_point_ms timestamp_start,
                                           const time_point_ms timestamp_end,
                                           std::chrono::milliseconds interval_ms)
{
    if (interval_ms.count() == 0 || interval_ms.count() >= INT_MAX || timestamp_start >= timestamp_end) {
        return std::make_shared<Intervals>(timestamp_start , interval_ms);
    }

    json11::Json command = json11::Json::object{
        { "command", "get_intervals" },
        { "exchange", exchange },
        { "symbol", symbol },
        { "timestamp_start", DateTime::to_string(timestamp_start) },
        { "timestamp_end", DateTime::to_string(timestamp_end) },
        { "interval_ms", (int) interval_ms.count() }
    };

    auto message = zmq::message_t{ command.dump() };
    auto result = client->send(message, zmq::send_flags::dontwait);
    
    const auto recv_result = client->recv(message);

    auto intervals_buffer = std::stringstream{ std::string(static_cast<char*>(message.data()), message.size()) };
    auto intervals = Intervals{ timestamp_start , interval_ms };
    intervals_buffer >> intervals;

    return std::make_shared<Intervals>(intervals);
}

sptrIntervals BitBaseClient::get_intervals(const std::string& symbol,
    const std::string& exchange,
    const time_point_ms timestamp_start,
    std::chrono::milliseconds interval_ms)
{
    const auto timestamp_end = system_clock_ms_now();
    return get_intervals(symbol, exchange, timestamp_start, timestamp_end, interval_ms);
}

sptrTicks BitBaseClient::get_ticks(const std::string& symbol,
    const std::string& exchange,
    const time_point_ms timestamp_start,
    const time_point_ms timestamp_end)
{
    json11::Json command = json11::Json::object{
        { "command", "get_ticks" },
        { "exchange", exchange },
        { "symbol", symbol },
        { "timestamp_start", DateTime::to_string(timestamp_start) },
        { "timestamp_end", DateTime::to_string(timestamp_end) }
    };

    auto message = zmq::message_t{ command.dump() };
    auto result = client->send(message, zmq::send_flags::dontwait);

    const auto recv_result = client->recv(message);

    auto ticks_buffer = std::stringstream{ std::string(static_cast<char*>(message.data()), message.size()) };
    auto ticks = Ticks{ };
    ticks_buffer >> ticks;

    return std::make_shared<Ticks>(ticks);
}
