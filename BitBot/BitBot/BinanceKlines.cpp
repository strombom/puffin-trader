#include "pch.h"

#include "BinanceKlines.h"


std::ostream& operator<<(std::ostream& stream, const BinanceKline& row)
{
    stream.write(reinterpret_cast<const char*>(&row.timestamp), sizeof(row.timestamp));
    stream.write(reinterpret_cast<const char*>(&row.open), sizeof(row.open));
    stream.write(reinterpret_cast<const char*>(&row.volume), sizeof(row.volume));

    return stream;
}

std::istream& operator>>(std::istream& stream, BinanceKline& row)
{
    stream.read(reinterpret_cast <char*> (&row.timestamp), sizeof(row.timestamp));
    stream.read(reinterpret_cast <char*> (&row.open), sizeof(row.open));
    stream.read(reinterpret_cast <char*> (&row.volume), sizeof(row.volume));

    return stream;
}

std::ostream& operator<<(std::ostream& stream, const BinanceKlines& binance_klines_data)
{
    for (auto&& row : binance_klines_data.rows) {
        stream << row;
    }

    return stream;
}

std::istream& operator>>(std::istream& stream, BinanceKlines& binance_klines_data)
{
    auto binance_kline = BinanceKline{};
    while (stream >> binance_kline) {
        binance_klines_data.rows.push_back(binance_kline);
    }

    return stream;
}

void BinanceKlines::load(const std::string& file_path)
{
    auto data_file = std::ifstream{ file_path, std::ios::binary };
    auto database_binance_kline = BinanceKline{};
    while (data_file >> database_binance_kline) {
        rows.push_back(database_binance_kline);
    }
    data_file.close();
}

void BinanceKlines::save(const std::string& file_path) const
{
    auto data_file = std::ofstream{ file_path, std::ios::binary };
    data_file << *this;
    data_file.close();
}

time_point_ms BinanceKlines::get_timestamp_begin(void) const
{
    return time_point_ms{}; // timestamp_start;
}

time_point_ms BinanceKlines::get_timestamp_end(void) const
{
    return time_point_ms{}; // timestamp_start + binance_kline * (rows.size() - 1);
}
