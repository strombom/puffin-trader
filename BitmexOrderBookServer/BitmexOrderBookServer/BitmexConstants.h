#pragma once

#include <array>

using namespace std::chrono_literals;

namespace Bitmex
{
    constexpr auto symbols = std::array<const char*, 3 > {"XBTUSD", "ETHUSD", "XRPUSD"};
    constexpr auto buffer_size = 5000000;
    constexpr auto server_address = "tcp://*:31005";
}
