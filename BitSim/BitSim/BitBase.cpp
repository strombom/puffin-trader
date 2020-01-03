
#include "BitBase.h"
#include "BitSimConstants.h"

#include "json.hpp"


BitBase::BitBase(void)
{
    client = std::make_unique<zmq::socket_t>(context, zmq::socket_type::req);
    client->connect(BitSim::BitBase::address);
}

void BitBase::get_intervals(void)
{
    constexpr auto timestamp_start = time_point_us{ date::sys_days(date::year{2019} / 06 / 01) };
    constexpr auto timestamp_end   = time_point_us{ date::sys_days(date::year{2020} / 01 / 01) };

    auto command = nlohmann::json{};
    command["command"] = "get_intervals";
    command["start"]   = timestamp_start.time_since_epoch().count();
    command["end"]     = timestamp_end.time_since_epoch().count();

    auto message = zmq::message_t{ command.dump() };
    auto result = client->send(message, zmq::send_flags::dontwait);
}
