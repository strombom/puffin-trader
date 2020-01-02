#pragma once

#include "DateTime.h"
#include "BitBaseConstants.h"

#include <vector>

using step_prices_t = std::array<float, BitBase::Interval::steps.size()>;

class DatabaseInterval
{
public:

    DatabaseInterval(float last_price, float vol_buy, float vol_sell, const step_prices_t& prices_buy, const step_prices_t& prices_sell) :
        last_price(last_price), vol_buy(vol_buy), vol_sell(vol_sell), prices_buy(prices_buy), prices_sell(prices_sell) {}

    friend std::ostream& operator<<(std::ostream& stream, const DatabaseInterval& row);

    float last_price;
    float vol_buy;
    float vol_sell;
    step_prices_t prices_buy;
    step_prices_t prices_sell;
};

class DatabaseIntervals
{
public:

    DatabaseIntervals(const time_point_us& timestamp_start, const std::chrono::seconds& interval) :
        timestamp_start(timestamp_start), interval(interval)
    {
        rows.reserve(BitBase::Interval::batch_size);
    }

    friend std::ostream& operator<<(std::ostream& stream, const DatabaseIntervals& intervals_data);

    std::vector<DatabaseInterval> rows;
    time_point_us timestamp_start;
    std::chrono::seconds interval;
};
