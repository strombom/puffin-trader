#pragma once

#include "DateTime.h"

#include <array>

using namespace std::chrono_literals;

namespace BitBase
{
    namespace Bitmex
    {
        constexpr auto exchange_name = "BITMEX"; 
        constexpr auto first_timestamp = time_point_us{ date::sys_days(date::year{2019} / 06 / 01) };

        namespace Daily
        {
            constexpr auto downloader_client_id = "bitmex_daily";
            constexpr auto base_url_start = "https://s3-eu-west-1.amazonaws.com/public.bitmex.com/data/trade/";
            constexpr auto base_url_end = ".csv.gz";
            constexpr auto url_date_format = "%Y%m%d";
            constexpr auto active_downloads_max = 5;
        }
    }

    namespace Interval
    {
        constexpr auto enabled_symbols = std::array<const char*, 1>{ "XBTUSD" };
        constexpr auto steps = std::array<float, 6>{ 1.0f, 2.0f, 5.0f, 10.0f, 20.0f, 50.0f };
        constexpr auto batch_timeout = 1s;
        constexpr auto batch_size = 10000;
        constexpr auto intervals = std::array<std::chrono::seconds, 1>{ 10s };
    }

    namespace Database
    {
        constexpr auto time_format = "%F %T";
        constexpr auto root_path = "C:\\development\\github\\puffin-trader\\database";
        constexpr auto sqlite_busy_timeout_ms = 5000;
    }
}

namespace BitSim
{
    namespace BitBase
    {
        constexpr auto address = "tcp://localhost:31000";
    }

    constexpr auto observations_path = "C:\\development\\github\\puffin-trader\\tmp\\observations.dat";

    constexpr auto n_batches = 20000;
    constexpr auto batch_size = 500;
    constexpr auto observation_length = 128;
    constexpr auto n_channels = 3;
    constexpr auto n_observations = 10;
    constexpr auto n_predictions = 1;
    constexpr auto n_positive = 1;
    constexpr auto n_negative = 19;

    constexpr auto feature_size = 128;

    const auto ch_price = 0;
    const auto ch_buy_volume = 1;
    const auto ch_sell_volume = 2;
}
