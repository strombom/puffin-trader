
#include "Logger.h"
#include "BitBaseClient.h"
#include "BitBotConstants.h"

#include "date/date.h"
#include "json11/json11.hpp"


BitBaseClient::BitBaseClient(void)
{
    client = std::make_unique<zmq::socket_t>(context, zmq::socket_type::req);
    client->connect(BitSim::BitBase::address);
}

void BitBaseClient::get_intervals(void)
{
    constexpr auto symbol          = "XBTUSD";
    constexpr auto exchange        = "BITMEX";
    constexpr auto timestamp_start = date::sys_days(date::year{ 2019 } / 06 / 01) + std::chrono::hours{ 0 } + std::chrono::minutes{ 0 };
    constexpr auto timestamp_end   = date::sys_days(date::year{2019} / 06 / 01) + std::chrono::seconds{ 30 };
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
    
    const auto recv_result = client->recv(message);

    const auto message_string = std::string(static_cast<char*>(message.data()), message.size());
    logger.info("BitBase::get_intervals msg: %d", message.size());



    printf("\n");
    printf("===\n");
    printf("\n");
    for (auto i = 0; i < message_string.size(); i++) {
        if (i % 16 == 0) {
            printf("\n");
        }
        printf("%02X ", (unsigned char) message_string[i]);
    }


}
