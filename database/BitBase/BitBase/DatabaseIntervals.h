#pragma once

#include "DateTime.h"
#include "BitBaseConstants.h"

#include <vector>

class DatabaseInterval
{
public:

    DatabaseInterval(float last_price, float vol_buy, float vol_sell, const BitBase::Interval::step_prices_t& prices_buy, const BitBase::Interval::step_prices_t& prices_sell) :
        last_price(last_price), vol_buy(vol_buy), vol_sell(vol_sell), prices_buy(prices_buy), prices_sell(prices_sell) {}

    friend std::ostream& operator<<(std::ostream& stream, const DatabaseInterval& row);

    float last_price;
    float vol_buy;
    float vol_sell;
    BitBase::Interval::step_prices_t prices_buy;
    BitBase::Interval::step_prices_t prices_sell;
};

class DatabaseIntervals
{
public:

    DatabaseIntervals(const time_point_us& timestamp_start, const std::chrono::seconds& interval) : timestamp_start(timestamp_start), interval(interval) {}

    friend std::ostream& operator<<(std::ostream& stream, const DatabaseIntervals& intervals_data);

    time_point_us get_timestamp_end(void) const;

    std::vector<DatabaseInterval> rows;
    time_point_us timestamp_start;
    std::chrono::seconds interval;
};
