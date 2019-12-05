#pragma once

#include "DateTime.h"
#include "BitBaseConstants.h"

#include <vector>

struct DatabaseTick
{
    DatabaseTick(void) {}

    DatabaseTick(const time_point_us timestamp, const float price, const float volume, const bool buy) :
        timestamp(timestamp), price(price), volume(volume), buy(buy) {}

    friend std::ostream& operator<<(std::ostream& stream, const DatabaseTick& row);
    friend std::istream& operator>>(std::istream& stream, DatabaseTick& row);

    time_point_us timestamp;
    float price;
    float volume;
    bool buy;
    
    static constexpr int struct_size = sizeof(timestamp) + sizeof(price) + sizeof(volume) + sizeof(buy);
};

using DatabaseTicks = std::vector<DatabaseTick>;

struct DatabaseInterval
{

    float last_price;
    float vol_buy;
    float vol_sell;
    std::array<float, BitBase::Interval::steps.size()> prices_buy;

    //constexpr auto steps = std::array<float, 6>{ 1.0f, 2.0f, 5.0f, 10.0f, 20.0f, 50.0f };
};


/*
class DatabaseTicks
{
public:
    DatabaseTicks(void);

    void append(const time_point_us timestamp, const float price, const float volume, const bool buy);
    size_t count(void);

    time_point_us get_first_timestamp(void);

    friend std::istream& operator>>(std::istream& stream, DatabaseTicks& row);

    std::vector<DatabaseTickRow> rows;
};
*/
