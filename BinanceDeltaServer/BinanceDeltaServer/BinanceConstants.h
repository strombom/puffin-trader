#pragma once

#include <array>

using namespace std::chrono_literals;

namespace Binance
{
    constexpr auto symbols = std::array<const char*, 2>{"BTCUSDT", "ETHUSDT"};
    constexpr auto buffer_size = 15000000;
    constexpr auto server_address = "tcp://*:31002";

    namespace websocket
    {
        constexpr auto host = "stream.binance.com";
        constexpr auto port = "9443";
        constexpr auto url = "/stream?streams=";
    }
}
