
#include "DatabaseTicks.h"


std::ostream& operator<<(std::ostream& stream, const DatabaseTick& row)
{
    const auto timestamp = row.timestamp.time_since_epoch().count();
    const auto price = row.price;
    const auto volume = row.volume;
    const auto buy = row.buy;

    stream.write(reinterpret_cast<const char*>(&timestamp), sizeof(timestamp));
    stream.write(reinterpret_cast<const char*>(&price), sizeof(price));
    stream.write(reinterpret_cast<const char*>(&volume), sizeof(volume));
    stream.write(reinterpret_cast<const char*>(&buy), sizeof(buy));

    return stream;
}

std::istream& operator>>(std::istream& stream, DatabaseTick& row)
{
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
