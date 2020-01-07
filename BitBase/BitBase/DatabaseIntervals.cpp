
#include "Logger.h"
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

std::istream& operator>>(std::istream& stream, DatabaseInterval& row)
{
    stream.read(reinterpret_cast <char*> (&row.last_price), sizeof(float));
    stream.read(reinterpret_cast <char*> (&row.vol_buy), sizeof(float));
    stream.read(reinterpret_cast <char*> (&row.vol_sell), sizeof(float));
    
    for (std::vector<step_prices_t>::size_type step = 0; step != row.prices_buy.size(); step++) {
        stream.read(reinterpret_cast <char*> (&row.prices_buy[step]), sizeof(float));
    }

    for (std::vector<step_prices_t>::size_type step = 0; step != row.prices_sell.size(); step++) {
        stream.read(reinterpret_cast <char*> (&row.prices_sell[step]), sizeof(float));
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

std::istream& operator>>(std::istream& stream, DatabaseIntervals& intervals_data)
{
    auto interval = DatabaseInterval{};
    while (stream >> interval) {
        intervals_data.rows.push_back(interval);
    }

    return stream;
}
