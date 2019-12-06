#pragma once

#include "DateTime.h"

#include <array>

using namespace std::chrono_literals;

namespace BitBase
{
    namespace Bitmex
    {
        constexpr auto exchange_name = "BITMEX";
        constexpr auto first_timestamp = time_point_us{ date::sys_days(date::year{2017} / 01 / 01) };

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
        constexpr auto batch_timeout = 1s;
        constexpr auto batch_size = 1000;
        constexpr auto intervals = std::array<std::chrono::seconds, 1>{ 10s };
        constexpr auto steps = std::array<float, 6>{ 1.0f, 2.0f, 5.0f, 10.0f, 20.0f, 50.0f };
    }

    namespace Database
    {
        constexpr auto time_format = "%F %T";
    }
}
