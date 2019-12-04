#pragma once

#include "DateTime.h"

#include <vector>

struct DatabaseTick
{
    DatabaseTick(const time_point_us timestamp, const float price, const float volume, const bool buy) :
        timestamp(timestamp), price(price), volume(volume), buy(buy) {}

    friend std::ostream& operator<<(std::ostream& stream, const DatabaseTick& row);
    friend std::istream& operator>>(std::istream& stream, DatabaseTick& row);

    const time_point_us timestamp;
    const float price;
    const float volume;
    const bool buy;
};

using DatabaseTicks = std::vector<DatabaseTick>;

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
