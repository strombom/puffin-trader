#pragma once

#include <chrono>

using namespace std::chrono_literals;

namespace Config {
    constexpr auto base_path = "E:/BitCounter/";

    namespace IntrinsicEvents
    {
        constexpr auto delta = 0.0005;
    }
}
