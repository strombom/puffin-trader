#include "pch.h"

#include "AggTicks.h"

#include <fstream>


std::ostream& operator<<(std::ostream& stream, const AggTick& row)
{
    const auto timestamp = row.timestamp.time_since_epoch().count();

    stream.write(reinterpret_cast<const char*>(&timestamp),  sizeof(timestamp));
    stream.write(reinterpret_cast<const char*>(&row.high),  sizeof(row.high));
    stream.write(reinterpret_cast<const char*>(&row.low), sizeof(row.low));
    stream.write(reinterpret_cast<const char*>(&row.volume), sizeof(row.volume));

    return stream;
}

std::istream& operator>>(std::istream& stream, AggTick& row)
{
    auto timestamp_ms = long long{};
    stream.read(reinterpret_cast<char*>(&timestamp_ms), sizeof(timestamp_ms));
    stream.read(reinterpret_cast<char*>(&row.high), sizeof(row.high));
    stream.read(reinterpret_cast<char*>(&row.low), sizeof(row.low));
    stream.read(reinterpret_cast<char*>(&row.volume),   sizeof(row.volume));

    row.timestamp = time_point_ms{ (std::chrono::milliseconds) timestamp_ms };

    return stream;
}

AggTicks::AggTicks(sptrTicks ticks)
{
    // Round timestamp
    auto agg_tick = AggTick{};
    auto agg_tick_valid = false;
    rows.clear();

    for (auto&& tick : ticks->rows) {
        if (agg_tick_valid && tick.timestamp >= agg_tick.timestamp + BitSim::aggregate) {
                rows.push_back(agg_tick);
                agg_tick_valid = false;
        }

        if (agg_tick_valid) {
            agg_tick.high = std::max(agg_tick.high, tick.price);
            agg_tick.low = std::min(agg_tick.low, tick.price);
            agg_tick.volume += tick.volume;
        }
        else {
            const auto agg_timestamp = (tick.timestamp.time_since_epoch() / BitSim::aggregate) * BitSim::aggregate;
            agg_tick = AggTick{ agg_timestamp,  tick.price, tick.price, tick.volume};
            agg_tick_valid = true;
        }
    }
}

AggTicks::AggTicks(const std::string filename_path)
{
    auto file = std::ifstream{ filename_path, std::ios::binary };

    auto agg_tick = AggTick{};
    while (file >> agg_tick) {
        rows.push_back(agg_tick);
    }

    file.close();
}

std::ostream& operator<<(std::ostream& stream, const AggTicks& agg_ticks)
{
    for (auto&& row : agg_ticks.rows) {
        stream << row;
    }

    return stream;
}

std::istream& operator>>(std::istream& stream, AggTicks& agg_ticks)
{
    auto tick = AggTick{};
    while (stream >> tick) {
        agg_ticks.rows.push_back(tick);
    }

    return stream;
}

void AggTicks::save(const std::string filename_path)
{
    auto file = std::ofstream{ filename_path, std::ofstream::binary };
    for (auto &&row : rows) {
        file << row;
    }
    file.close();
}
