#pragma once

#include <array>
#include <chrono>


using namespace std::literals::chrono_literals;

namespace Coinbase
{
    constexpr auto symbols = std::array<const char*, 2>{"BTC-USD"};
    constexpr auto buffer_size = 100000;
    constexpr auto server_address = "tcp://*:31004";

    namespace RestApi
    {
        constexpr auto api_key = "rsjxKT5NPu3JWEaJ";
        constexpr auto api_secret = "VAfCgAjQe9RCJiwVUynlCqGd56QLPbg6";
        constexpr auto rest_api_auth_timeout = 60s;
        constexpr auto rest_api_host = "api.pro.coinbase.com";
        constexpr auto rest_api_port = "443";
        constexpr auto rest_api_url = "/";

        constexpr auto rate_limit = 250ms;
        constexpr auto max_rows = 100;
    }

    namespace websocket
    {
        constexpr auto host = "stream.coinbase.com";
        constexpr auto port = "9443";
        constexpr auto url = "/stream?streams=";
    }
}
