#include "pch.h"

#include "Ticks.h"

#include <fstream>


std::ostream& operator<<(std::ostream& stream, const Tick& row)
{
    const auto timestamp = row.timestamp.time_since_epoch().count();

    stream.write(reinterpret_cast<const char*>(&timestamp),  sizeof(timestamp));
    stream.write(reinterpret_cast<const char*>(&row.price),  sizeof(row.price));
    stream.write(reinterpret_cast<const char*>(&row.volume), sizeof(row.volume));
    stream.write(reinterpret_cast<const char*>(&row.buy),    sizeof(row.buy));

    return stream;
}

std::istream& operator>>(std::istream& stream, Tick& row)
{
    auto timestamp_ms = long long{};
    stream.read(reinterpret_cast<char*>(&timestamp_ms), sizeof(timestamp_ms));
    stream.read(reinterpret_cast<char*>(&row.price),    sizeof(row.price));
    stream.read(reinterpret_cast<char*>(&row.volume),   sizeof(row.volume));
    stream.read(reinterpret_cast<char*>(&row.buy),      sizeof(row.buy));

    row.timestamp = time_point_ms{ (std::chrono::milliseconds) timestamp_ms };

    return stream;
}

Ticks::Ticks(const std::string filename_path)
{
    auto file = std::ifstream{ filename_path, std::ios::binary };

    auto tick = Tick{};
    while (file >> tick) {
        rows.push_back(tick);
    }

    file.close();
}

std::ostream& operator<<(std::ostream& stream, const Ticks& ticks_data)
{
    for (auto&& row : ticks_data.rows) {
        stream << row;
    }

    return stream;
}

std::istream& operator>>(std::istream& stream, Ticks& ticks_data)
{
    auto tick = Tick{};
    while (stream >> tick) {
        ticks_data.rows.push_back(tick);
    }

    return stream;
}

void Ticks::save(const std::string filename_path)
{
    auto file = std::ofstream{ filename_path, std::ofstream::binary };
    for (auto &&row : rows) {
        file << row;
    }
    file.close();
}
