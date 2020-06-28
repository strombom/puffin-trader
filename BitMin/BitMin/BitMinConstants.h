#pragma once

#include "BitLib/DateTime.h"

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
        constexpr auto static_path = "/home/strombom/bitmin_static";
        constexpr auto port = 8443;
        constexpr auto address = "0.0.0.0";
    }
}
