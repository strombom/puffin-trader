#pragma once

#include <array>
#include <chrono>


using namespace std::chrono_literals;

namespace CoinbasePro
{
    constexpr auto symbols = std::array<const char*, 2>{"BTC-USD", "ETH-USD"};
    constexpr auto buffer_size = 1000000;
    constexpr auto server_address = "tcp://*:31004";

    namespace WebSocket
    {
        constexpr auto host = "ws-feed.pro.coinbase.com";
        constexpr auto port = "443";
        constexpr auto url = "/";
    }
}
