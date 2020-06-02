#pragma once

#include <array>


namespace Bitmex
{
    constexpr auto symbols = std::array<const char*, 1>{"XBTUSD"}; //, "ETHUSD", "XRPUSD"};
    constexpr auto buffer_size = 10000;
    constexpr auto server_address = "tcp://*:31002";
}
