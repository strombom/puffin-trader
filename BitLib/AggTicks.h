#pragma once
#include "pch.h"

#include "BitLib/Ticks.h"
#include "BitLib/DateTime.h"

#include <vector>


class AggTick
{
public:
    AggTick(void) : high(0), low(0), volume(0) {}

    AggTick(const time_point_ms timestamp, const float high, const float low, const float volume) :
        timestamp(timestamp), high(high), low(low), volume(volume) {}

    friend std::ostream& operator<<(std::ostream& stream, const AggTick& row);
    friend std::istream& operator>>(std::istream& stream, AggTick& row);

    time_point_ms timestamp;
    float high;
    float low;
    float volume;
    
    static constexpr int struct_size = sizeof(timestamp) + sizeof(high) + sizeof(low) + sizeof(volume);
};

class AggTicks
{
public:
    AggTicks(void) {}
    AggTicks(sptrTicks ticks);
    AggTicks(const std::string filename_path);

    std::vector<AggTick> rows;

    friend std::ostream& operator<<(std::ostream& stream, const AggTicks& agg_ticks_data);
    friend std::istream& operator>>(std::istream& stream, AggTicks& agg_ticks_data);

    void save(const std::string filename_path);
};

using sptrAggTicks = std::shared_ptr<AggTicks>;
