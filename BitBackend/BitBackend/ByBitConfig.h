#pragma once

#include <chrono>
#include <map>

using namespace std::chrono_literals;

namespace ByBit {

    namespace WebSocket {
        constexpr auto host = "stream-testnet.bybit.com";
        constexpr auto port = "443";
        constexpr auto auth_timeout = 60s;
        constexpr auto url_private = "/realtime_private";
        constexpr auto url_public = "/realtime_public";
        constexpr auto reconnect_delay = 2s;
    }

    namespace Rest {
        constexpr auto base_endpoint = "https://api-testnet.bybit.com";
        constexpr auto host = "api-testnet.bybit.com";
        constexpr auto service = "https";

        enum class Endpoint : int {
            heartbeat_ping,
            position_list,
            create_order,
            cancel_order,
            cancel_all_orders
        };

        constexpr auto endpoints = std::array<const char*, 5>{
            "/public/linear/recent-trading-records",
            "/private/linear/position/list",
            "/private/linear/order/create",
            "/private/linear/order/cancel",
            "/private/linear/order/cancel-all"
        };
    }

    constexpr auto api_key = "rvfw51waQa270JP0Rn";
    constexpr auto api_secret = "h1zoAnp7ZWobI4tkeN6d1hNOktIEfN085Hyb";
    //constexpr auto api_key = "XQH3HEpdgNNqY0NAgx";
    //constexpr auto api_secret = "QM0PoC9OG3v6zh7oKEjeJI0EIyeezo8gEIfZ";
}
