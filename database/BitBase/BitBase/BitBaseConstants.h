#pragma once

#include "DateTime.h"

#include <array>

using namespace std::chrono_literals;

namespace BitBase
{
    namespace Bitmex
    {
        static constexpr auto exchange_name = "BITMEX";
        static constexpr auto first_timestamp = time_point_us{ date::sys_days(date::year{2017} / 01 / 01) };

        namespace Daily
        {
            static constexpr auto downloader_client_id = "bitmex_daily";
            static constexpr auto base_url_start = "https://s3-eu-west-1.amazonaws.com/public.bitmex.com/data/trade/";
            static constexpr auto base_url_end = ".csv.gz";
            static constexpr auto url_date_format = "%Y%m%d";
            static constexpr auto active_downloads_max = 5;
        }
    }

    namespace Interval
    {
        static const auto batch_size = 100;
        static constexpr auto intervals = std::array<std::chrono::seconds, 1>{120s};
        constexpr auto steps = std::array<float, 6>{ 1.0f, 2.0f, 5.0f, 10.0f, 20.0f, 50.0f };
    }
}
