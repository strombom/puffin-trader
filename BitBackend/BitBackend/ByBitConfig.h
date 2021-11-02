#pragma once

#include <chrono>

using namespace std::chrono_literals;

namespace ByBit {


    namespace websocket {
        constexpr auto host = "stream.bybit.com";
        constexpr auto port = "443";
        constexpr auto auth_timeout = 60s;
        constexpr auto url_private = "/realtime_private";
        constexpr auto url_public = "/realtime_public";
        constexpr auto reconnect_delay = 2s;
    }

    constexpr auto api_key = "XQH3HEpdgNNqY0NAgx";
    constexpr auto api_secret = "QM0PoC9OG3v6zh7oKEjeJI0EIyeezo8gEIfZ";
}
