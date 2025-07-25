#pragma once
#include "pch.h"

#include "DateTime.h"
#include "BitBotConstants.h"

#include <array>
#include <vector>


//using step_prices_t = std::array<float, BitBase::Bitmex::Interval::steps.size()>;

class Interval
{
public:
    Interval(void) : last_price(0), vol_buy(0), vol_sell(0) {}
    Interval(float last_price, float vol_buy, float vol_sell) : //, const step_prices_t& prices_buy, const step_prices_t& prices_sell) :
        last_price(last_price), vol_buy(vol_buy), vol_sell(vol_sell) {} //, prices_buy(prices_buy), prices_sell(prices_sell) {}

    friend std::ostream& operator<<(std::ostream& stream, const Interval& row);
    friend std::istream& operator>>(std::istream& stream, Interval& row);

    float last_price;
    float vol_buy;
    float vol_sell;
    //step_prices_t prices_buy{ 0 };
    //step_prices_t prices_sell{ 0 };
};

class Intervals
{
public:
    Intervals(const time_point_ms& timestamp_start, const std::chrono::milliseconds& interval) :
        timestamp_start(timestamp_start), interval(interval) {}

    Intervals(const std::string& file_path)
    {
        load(file_path);
    }

    // Copy constructor
    Intervals(const Intervals& intervals) :
        rows(intervals.rows), timestamp_start(intervals.timestamp_start), interval(intervals.interval) {}

    void rotate_insert(const std::shared_ptr<const Intervals> intervals);

    void load(const std::string& file_path);
    void save(const std::string& file_path) const;

    time_point_ms get_timestamp_start(void) const;
    time_point_ms get_timestamp_last(void) const;

    friend std::ostream& operator<<(std::ostream& stream, const Intervals& intervals_data);
    friend std::istream& operator>>(std::istream& stream, Intervals& intervals_data);

    std::vector<Interval> rows;
    time_point_ms timestamp_start;
    std::chrono::milliseconds interval;
};

using sptrIntervals = std::shared_ptr<Intervals>;
