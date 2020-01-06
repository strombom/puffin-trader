
#include "Logger.h"
#include "BitBase.h"
#include "BitSimConstants.h"

#include "date/date.h"
#include "json11/json11.hpp"


BitBase::BitBase(void)
{
    client = std::make_unique<zmq::socket_t>(context, zmq::socket_type::req);
    client->connect(BitSim::BitBase::address);
}

void BitBase::get_intervals(void)
{
    constexpr auto symbol          = "XBTUSD";
    constexpr auto exchange        = "BITMEX";
    constexpr auto timestamp_start = date::sys_days(date::year{2019} / 07 / 01);
    constexpr auto timestamp_end   = date::sys_days(date::year{2019} / 07 / 02);
    constexpr auto interval        = std::chrono::seconds{ 10s };

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
}
