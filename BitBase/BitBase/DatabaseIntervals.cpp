#include "DatabaseIntervals.h"


std::ostream& operator<<(std::ostream& stream, const DatabaseInterval& row)
{
    stream.write(reinterpret_cast<const char*>(&row.last_price), sizeof(row.last_price));
    stream.write(reinterpret_cast<const char*>(&row.vol_buy), sizeof(row.vol_buy));
    stream.write(reinterpret_cast<const char*>(&row.vol_sell), sizeof(row.vol_sell));

    for (auto&& step : row.prices_buy) {
        stream.write(reinterpret_cast<const char*>(&step), sizeof(step));
    }

    for (auto&& step : row.prices_sell) {
        stream.write(reinterpret_cast<const char*>(&step), sizeof(step));
    }

    return stream;
}

std::ostream& operator<<(std::ostream& stream, const DatabaseIntervals& intervals_data)
{
    for (auto&& row : intervals_data.rows) {
        stream << row;
    }

    return stream;
}


