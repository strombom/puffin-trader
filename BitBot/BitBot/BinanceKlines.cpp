#include "pch.h"

#include "BinanceKlines.h"
#include "BitLib/BitBotConstants.h"

#include <fstream>
#include <filesystem>


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

BinanceKlines::BinanceKlines(const std::string& symbol) : symbol(symbol)
{
    load();
}

BinanceKlines::BinanceKlines(const std::string& symbol, time_point_ms begin) : symbol(symbol)
{
    load(begin);
}

void BinanceKlines::load(void)
{
    load(date::sys_days(date::year{ 2020 } / 1 / 1) + 0h + 0min + 0s);
}

void BinanceKlines::load(time_point_ms begin)
{
    const auto file_path = std::string{ BitBot::path } + "/klines/" + symbol + ".dat";

    if (std::filesystem::exists(file_path)) {
        auto data_file = std::ifstream{ file_path, std::ios::binary };
        auto database_binance_kline = BinanceKline{};
        auto started = false;
        while (data_file >> database_binance_kline) {
            if (!started && database_binance_kline.timestamp >= begin) {
                started = true;
            }
            if (started) {
                rows.push_back(database_binance_kline);
            }            
        }
        data_file.close();
    }
}

void BinanceKlines::save(void) const
{
    const auto directory = std::string{ BitBot::path } + "/klines/";
    std::filesystem::create_directory(directory);
    const auto file_path = directory + symbol + ".dat";
    auto data_file = std::ofstream{ file_path, std::ios::binary };
    data_file << *this;
    data_file.close();
}

time_point_ms BinanceKlines::get_timestamp_begin(void) const
{
    return rows.front().timestamp;
}

time_point_ms BinanceKlines::get_timestamp_end(void) const
{
    return rows.back().timestamp;
}
