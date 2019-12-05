
#include "DatabaseTicks.h"

/*
constexpr int DatabaseTick::get_struct_size(void)
{
    return sizeof(timestamp) + sizeof(price) + sizeof(volume) + sizeof(buy);
}
*/

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

/*
DatabaseTicks::DatabaseTicks(void)
{

}
*/

//std::istream& operator>>(std::istream& stream, DatabaseTicks& row)
//{
    /*
    const auto timestamp = row.timestamp.time_since_epoch().count();
    const auto price = row.price;
    const auto volume = row.volume;
    const auto buy = row.buy;

    stream.write(reinterpret_cast<const char*>(&timestamp), sizeof(timestamp));
    stream.write(reinterpret_cast<const char*>(&price), sizeof(price));
    stream.write(reinterpret_cast<const char*>(&volume), sizeof(volume));
    stream.write(reinterpret_cast<const char*>(&buy), sizeof(buy));

    return stream;
    */

    //return stream;
//}

/*
void DatabaseTicks::append(const time_point_us timestamp, const float price, const float volume, const bool buy)
{
    rows.push_back(DatabaseTickRow(timestamp, price, volume, buy));
}

time_point_us DatabaseTicks::get_first_timestamp(void)
{
    return rows[0].timestamp;
}

size_t DatabaseTicks::count(void)
{
    return rows.size();
}
*/
