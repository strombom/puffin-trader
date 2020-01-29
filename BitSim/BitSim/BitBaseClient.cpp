
#include "Logger.h"
#include "BitBaseClient.h"
#include "BitBotConstants.h"

#include "json11/json11.hpp"


BitBaseClient::BitBaseClient(void)
{
    client = std::make_unique<zmq::socket_t>(context, zmq::socket_type::req);
    client->connect(BitSim::BitBase::address);
}

sptrIntervals BitBaseClient::get_intervals(const std::string& symbol,
                                           const std::string& exchange,
                                           const time_point_s timestamp_start,
                                           const time_point_s timestamp_end,
                                           std::chrono::seconds interval)
{

    assert(interval.count() > 0 && interval.count() <= INT_MAX);
    assert(timestamp_end > timestamp_start);

    json11::Json command = json11::Json::object{
        { "command", "get_intervals" },
        { "exchange", exchange },
        { "symbol", symbol },
        { "timestamp_start", DateTime::to_string(timestamp_start) },
        { "timestamp_end", DateTime::to_string(timestamp_end) },
        { "interval_seconds", (int) interval.count() }
    };

    auto message = zmq::message_t{ command.dump() };
    auto result = client->send(message, zmq::send_flags::dontwait);
    
    const auto recv_result = client->recv(message);

    auto intervals_buffer = std::stringstream{ std::string(static_cast<char*>(message.data()), message.size()) };
    auto intervals = Intervals{ timestamp_start , interval };
    intervals_buffer >> intervals;

    return std::make_shared<Intervals>(intervals);
}
