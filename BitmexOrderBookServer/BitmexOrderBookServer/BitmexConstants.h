#pragma once

#include <array>

using namespace std::chrono_literals;

namespace Bitmex
{
    constexpr auto symbols = std::array<const char*, 1>{"XBTUSD"}; // , "ETHUSD", "XRPUSD"
    constexpr auto buffer_size = 365*24*60*60;
    constexpr auto server_address = "tcp://*:31005";
}
