#pragma once
#include "pch.h"

#include "BitLib/Ticks.h"
#include "BitLib/DateTime.h"
#include "BitLib/BitBotConstants.h"

#include <vector>


class AggTick
{
public:
    AggTick(void) : 
        timestamp(0ms),
        high(std::numeric_limits<float>::min()), 
        low(std::numeric_limits<float>::max()),
        volume(0) {}

    AggTick(const std::chrono::milliseconds timestamp) :
        timestamp(std::chrono::time_point<std::chrono::system_clock, std::chrono::milliseconds>{ timestamp }),
        high(std::numeric_limits<float>::min()),
        low(std::numeric_limits<float>::max()),
        volume(0) {}

    AggTick(const time_point_ms timestamp, const float high, const float low, const float volume) :
        timestamp(timestamp), high(high), low(low), volume(volume) {}

    AggTick(const std::chrono::milliseconds timestamp, const float high, const float low, const float volume) :
        timestamp(std::chrono::time_point<std::chrono::system_clock, std::chrono::milliseconds>{ timestamp }), high(high), low(low), volume(volume) {}

    friend std::ostream& operator<<(std::ostream& stream, const AggTick& agg_tick);
    friend std::istream& operator>>(std::istream& stream, AggTick& agg_tick);

    time_point_ms timestamp;
    float high;
    float low;
    float volume;
    
    static constexpr int struct_size = sizeof(timestamp) + sizeof(high) + sizeof(low) + sizeof(volume);
};

using sptrAggTick = std::shared_ptr<AggTick>;

class AggTicks
{
public:
    AggTicks(void);
    AggTicks(const sptrTicks ticks);
    AggTicks(const std::string filename_path);

    std::vector<AggTick> agg_ticks;

    AggTick pending_agg_tick;
    bool pending_agg_tick_valid;

    friend std::ostream& operator<<(std::ostream& stream, const AggTicks& agg_ticks_data);
    friend std::istream& operator>>(std::istream& stream, AggTicks& agg_ticks_data);

    void save(const std::string& filename_path) const;
    void load(const std::string& filename_path);

    void insert(const Tick& tick);
    void insert(const sptrTicks ticks);
};

using sptrAggTicks = std::shared_ptr<AggTicks>;
