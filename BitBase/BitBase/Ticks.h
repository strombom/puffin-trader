#pragma once

#include "DateTime.h"
#include "BitBaseConstants.h"

#include <vector>

class DatabaseTick
{
public:

    DatabaseTick(void) : price(0), volume(0), buy(0) {}

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

class DatabaseTicks
{
public:

    std::vector<DatabaseTick> rows;
};

using uptrDatabaseTicks = std::unique_ptr<DatabaseTicks>;
