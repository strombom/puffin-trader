
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
    
    json11::Json command = json11::Json::object{
        { "command", "get_intervals" },
        { "start", date::format("%F %T", timestamp_start) },
        { "end", date::format("%F %T", timestamp_end) },
    };

    auto message = zmq::message_t{ command.dump() };
    auto result = client->send(message, zmq::send_flags::dontwait);
}
