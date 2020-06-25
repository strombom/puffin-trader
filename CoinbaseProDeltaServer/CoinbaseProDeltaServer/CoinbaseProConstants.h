#pragma once

#include <array>
#include <chrono>


using namespace std::chrono_literals;

namespace CoinbasePro
{
    constexpr auto symbols = std::array<const char*, 1>{"BTC-USD"};
    constexpr auto buffer_size = 100000;
    constexpr auto server_address = "tcp://*:31004";

    namespace RestApi
    {
        constexpr auto api_key = "a05c7473808301946fc2aefe6a2adf08";
        constexpr auto api_secret = "lb/jwhlBh3wx7iQYtFlgZRkJiLviSZo0ZyuNTAbB9X8tCOAkN2uzrJwi1v+LcsO1InAgpBBrQmyd0WL3MUEu2A==";
        constexpr auto api_passphrase = "qk07ka5fj5q";
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
