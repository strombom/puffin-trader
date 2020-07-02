#include "pch.h"
#include "DirectionClient.h"


DirectionClient::DirectionClient(void)
{
    client = std::make_unique<zmq::socket_t>(context, zmq::socket_type::req);
    client->connect("tcp://localhost:31005");
}

void DirectionClient::send(std::map<std::string, sptrIntervals> prices, std::vector<int> directions)
{
    auto exchange_prices = std::map<std::string, std::vector<double>>{};
    for (const auto [exchange_name, intervals] : prices) {
        exchange_prices.insert({ exchange_name, std::vector<double>{} });

        for (auto idx = 0; idx < intervals->rows.size(); ++idx) {

            exchange_prices[exchange_name].push_back((double) intervals->rows[idx].last_price);
        }
    }

    json11::Json send_command = json11::Json::object{
        { "command", "set_directions" },
        { "prices", exchange_prices },
        { "directions", directions },
        { "names", json11::Json::object{
            {"bitmex", "Bitmex"},
            {"binance", "Binance"},
            {"coinbase", "Coinbase"}
        }}
    };

    auto message = zmq::message_t{ send_command.dump() };
    auto result = client->send(message, zmq::send_flags::dontwait);

    const auto recv_result = client->recv(message);

    const auto message_string = std::string(static_cast<char*>(message.data()), message.size());

    auto error_message = std::string{ "{\"command\":\"error\"}" };
    const auto rcv_command = json11::Json::parse(message_string.c_str(), error_message);
    const auto command_name = rcv_command["command"].string_value();
}
