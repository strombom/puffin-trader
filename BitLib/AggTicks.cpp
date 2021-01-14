#include "pch.h"

#include "AggTicks.h"

#include <fstream>


std::ostream& operator<<(std::ostream& stream, const AggTick& agg_tick)
{
    const auto timestamp = agg_tick.timestamp.time_since_epoch().count();

    stream.write(reinterpret_cast<const char*>(&timestamp),  sizeof(timestamp));
    stream.write(reinterpret_cast<const char*>(&agg_tick.high),  sizeof(agg_tick.high));
    stream.write(reinterpret_cast<const char*>(&agg_tick.low), sizeof(agg_tick.low));
    stream.write(reinterpret_cast<const char*>(&agg_tick.volume), sizeof(agg_tick.volume));

    return stream;
}

std::istream& operator>>(std::istream& stream, AggTick& agg_tick)
{
    auto timestamp_ms = long long{};
    stream.read(reinterpret_cast<char*>(&timestamp_ms), sizeof(timestamp_ms));
    stream.read(reinterpret_cast<char*>(&agg_tick.high), sizeof(agg_tick.high));
    stream.read(reinterpret_cast<char*>(&agg_tick.low), sizeof(agg_tick.low));
    stream.read(reinterpret_cast<char*>(&agg_tick.volume),   sizeof(agg_tick.volume));

    agg_tick.timestamp = time_point_ms{ (std::chrono::milliseconds) timestamp_ms };

    return stream;
}

AggTicks::AggTicks(void) :
    pending_agg_tick_valid(false)
{

}

AggTicks::AggTicks(const sptrTicks ticks) :
    pending_agg_tick_valid(false)
{
    insert(ticks);
}

void AggTicks::insert(const sptrTicks ticks)
{
    for (auto&& tick : ticks->rows) {
        insert(tick);
    }
}

void AggTicks::insert(const Tick& tick)
{
    if (pending_agg_tick_valid && tick.timestamp >= pending_agg_tick.timestamp + BitSim::aggregate) {
        agg_ticks.push_back(pending_agg_tick);
        pending_agg_tick_valid = false;
    }

    if (pending_agg_tick_valid) {
        pending_agg_tick.high = std::max(pending_agg_tick.high, tick.price);
        pending_agg_tick.low = std::min(pending_agg_tick.low, tick.price);
        pending_agg_tick.volume += tick.volume;
    }
    else {
        const auto agg_timestamp = (tick.timestamp.time_since_epoch() / BitSim::aggregate) * BitSim::aggregate;
        pending_agg_tick = AggTick{ agg_timestamp,  tick.price, tick.price, tick.volume };
        pending_agg_tick_valid = true;
    }
}

AggTicks::AggTicks(const std::string filename_path) :
    pending_agg_tick_valid(false)
{
    auto file = std::ifstream{ filename_path, std::ios::binary };

    auto agg_tick = AggTick{};
    while (file >> agg_tick) {
        agg_ticks.push_back(agg_tick);
    }

    file.close();
}

std::ostream& operator<<(std::ostream& stream, const AggTicks& agg_ticks)
{
    for (auto&& row : agg_ticks.agg_ticks) {
        stream << row;
    }

    return stream;
}

std::istream& operator>>(std::istream& stream, AggTicks& agg_ticks)
{
    auto tick = AggTick{};
    while (stream >> tick) {
        agg_ticks.agg_ticks.push_back(tick);
    }

    return stream;
}

void AggTicks::save(const std::string& filename_path) const
{
    auto file = std::ofstream{ filename_path, std::ofstream::binary };
    for (auto && agg_tick : agg_ticks) {
        file << agg_tick;
    }
    file.close();
}

void AggTicks::load(const std::string& filename_path)
{
    auto data_file = std::ifstream{ filename_path, std::ios::binary };
    auto agg_tick = AggTick{};
    while (data_file >> agg_tick) {
        agg_ticks.push_back(agg_tick);
    }
    data_file.close();
}
