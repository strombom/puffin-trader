#pragma once

#include "DateTime.h"

#include <array>

using namespace std::chrono_literals;

namespace BitMin
{
    namespace Bitmex
    {
        constexpr auto exchange_name = "BITMEX";
    }

    namespace HttpServer
    {
        constexpr auto static_path = "C:\\development\\github\\puffin-trader\\BitMin\\static\\";
        constexpr auto port = 8080;
        constexpr auto address = "0.0.0.0"; // localhost";
    }
}
