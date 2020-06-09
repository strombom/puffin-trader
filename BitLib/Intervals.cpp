#include "pch.h"

#include "Logger.h"
#include "Intervals.h"


std::ostream& operator<<(std::ostream& stream, const Interval& row)
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

std::istream& operator>>(std::istream& stream, Interval& row)
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

std::ostream& operator<<(std::ostream& stream, const Intervals& intervals_data)
{
    for (auto&& row : intervals_data.rows) {
        stream << row;
    }

    return stream;
}

std::istream& operator>>(std::istream& stream, Intervals& intervals_data)
{
    auto interval = Interval{};
    while (stream >> interval) {
        intervals_data.rows.push_back(interval);
    }

    return stream;
}

void Intervals::load(const std::string& file_path)
{
    auto start_time_raw = 0;
    auto interval_raw = 0;

    auto attr_file = std::ifstream{ file_path + "_attr" };
    attr_file >> start_time_raw >> interval_raw;
    attr_file.close();

    timestamp_start = time_point_ms{ std::chrono::milliseconds{start_time_raw} };
    interval = std::chrono::milliseconds{ interval_raw };

    auto data_file = std::ifstream{ file_path + "_data", std::ios::binary };
    auto database_interval = Interval{};
    while (data_file >> database_interval) {
        rows.push_back(database_interval);
    }
    data_file.close();
}

void Intervals::save(const std::string& file_path) const
{
    auto data_file = std::ofstream{ file_path + "_data", std::ios::binary };
    data_file << *this;
    data_file.close();

    auto attr_file = std::ofstream{ file_path + "_attr" };
    attr_file << timestamp_start.time_since_epoch().count() << std::endl;
    attr_file << interval.count() << std::endl;
    attr_file.close();
}

time_point_ms Intervals::get_timestamp_start(void) const
{
    return timestamp_start;
}

time_point_ms Intervals::get_timestamp_last(void) const
{
    return timestamp_start + interval * (rows.size() - 1);
}
