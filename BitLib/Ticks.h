#pragma once
#include "pch.h"

#include "BitLib/DateTime.h"

#include <vector>


class Tick
{
public:
    Tick(void) : price(0), volume(0), buy(0) {}

    Tick(const time_point_ms timestamp, const float price, const float volume, const bool buy) :
        timestamp(timestamp), price(price), volume(volume), buy(buy) {}

    friend std::ostream& operator<<(std::ostream& stream, const Tick& row);
    friend std::istream& operator>>(std::istream& stream, Tick& row);

    time_point_ms timestamp;
    float price;
    float volume;
    bool buy;
    
    static constexpr int struct_size = sizeof(timestamp) + sizeof(price) + sizeof(volume) + sizeof(buy);
};

class Ticks
{
public:
    Ticks(void) {}
    Ticks(const std::string filename_path);

    std::vector<Tick> rows;

    friend std::ostream& operator<<(std::ostream& stream, const Ticks& ticks_data);
    friend std::istream& operator>>(std::istream& stream, Ticks& ticks_data);

    void save(const std::string filename_path);
};

using sptrTicks = std::shared_ptr<Ticks>;
