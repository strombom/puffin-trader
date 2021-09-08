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
    //load_test_klines();
}

void BinanceKlines::load_test_klines(void)
{
    const auto file_path = std::string{ BitBotLiveV1::path } + std::string{ "/test_klines.csv" };

    auto file = std::ifstream{ file_path };

    auto symbol_col_idx = 0;
    auto found_symbol = false;

    if (file.good()) {
        auto line = std::string{};
        std::getline(file, line);
        auto ss = std::stringstream{ line };

        auto colname = std::string{};
        while (std::getline(ss, colname, ',')) {
            if (colname == symbol) {
                found_symbol = true;
                break;
            }
            symbol_col_idx++;
        }
    }

    if (!found_symbol) {
        throw std::exception();
    }

    auto timestamp = time_point_ms{ date::sys_days(date::year{2021} / 1 / 1) + std::chrono::hours{ 0 } };
    auto line = std::string{};
    while (std::getline(file, line)) {
        auto col_idx = 0;

        auto ss = std::stringstream{ line };
        auto price_string = std::string{};
        while (std::getline(ss, price_string, ',')) {
            if (col_idx == symbol_col_idx) {
                auto price = std::stof(price_string);
                rows.push_back({ timestamp, price, 0 });
                timestamp += std::chrono::minutes{ 1 };
                break;
            }
            col_idx++;
        }
    }
}

void BinanceKlines::load(void)
{
    const auto file_path = std::string{ BitBotLiveV1::path } + "/klines/" + symbol + ".dat";

    if (std::filesystem::exists(file_path)) {
        auto data_file = std::ifstream{ file_path, std::ios::binary };
        auto database_binance_kline = BinanceKline{};
        while (data_file >> database_binance_kline) {
            rows.push_back(database_binance_kline);
        }
        data_file.close();
    }
}

void BinanceKlines::save(void) const
{
    const auto directory = std::string{ BitBotLiveV1::path } + "/klines/";
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
