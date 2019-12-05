#pragma once

#include "DateTime.h"

namespace BitbaseConstants
{
    static constexpr auto bitmex_exchange_name = "BITMEX";
    static constexpr auto bitmex_first_timestamp = time_point_us{ date::sys_days(date::year{2017} / 01 / 01) };
}
