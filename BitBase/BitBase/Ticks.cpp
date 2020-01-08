
#include "Ticks.h"


std::ostream& operator<<(std::ostream& stream, const DatabaseTick& row)
{
    const auto timestamp = row.timestamp.time_since_epoch().count();

    stream.write(reinterpret_cast<const char*>(&timestamp),  sizeof(timestamp));
    stream.write(reinterpret_cast<const char*>(&row.price),  sizeof(row.price));
    stream.write(reinterpret_cast<const char*>(&row.volume), sizeof(row.volume));
    stream.write(reinterpret_cast<const char*>(&row.buy),    sizeof(row.buy));

    return stream;
}

std::istream& operator>>(std::istream& stream, DatabaseTick& row)
{
    auto timestamp_us = long long{};
    stream.read(reinterpret_cast<char*>(&timestamp_us), sizeof(timestamp_us));
    stream.read(reinterpret_cast<char*>(&row.price),    sizeof(row.price));
    stream.read(reinterpret_cast<char*>(&row.volume),   sizeof(row.volume));
    stream.read(reinterpret_cast<char*>(&row.buy),      sizeof(row.buy));

    row.timestamp = time_point_us{ (std::chrono::microseconds) timestamp_us };

    return stream;
}
