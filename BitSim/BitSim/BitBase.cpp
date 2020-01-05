
#include "Logger.h"
#include "BitBase.h"
#include "BitSimConstants.h"

#include "date.h"
#include "json11.hpp"


BitBase::BitBase(void)
{
    client = std::make_unique<zmq::socket_t>(context, zmq::socket_type::req);
    client->connect(BitSim::BitBase::address);
}

void BitBase::get_intervals(void)
{
    constexpr auto timestamp_start = date::sys_days(date::year{2019} / 06 / 01);
    constexpr auto timestamp_end   = date::sys_days(date::year{2020} / 01 / 01);
    constexpr auto symbol          = "XBTUSD";
    
    json11::Json command = json11::Json::object{
        { "command", "get_intervals" },
        { "symbol", symbol },
        { "timestamp_start", DateTime::to_string(timestamp_start) },
        { "timestamp_end", DateTime::to_string(timestamp_end) },
    };

    auto message = zmq::message_t{ command.dump() };
    auto result = client->send(message, zmq::send_flags::dontwait);
}
