#include "pch.h"
#include "Klines.h"
#include "BitLib/BitBotConstants.h"

#include <fstream>


Klines::Klines()
{
    for (const auto& symbol : BitBot::symbols) {
        printf("Load klines %s\n", symbol);
        load(symbol);
        break;
    }
}

void Klines::save(const std::string symbol)
{

}

bool Klines::load(const std::string symbol)
{
    const auto file_path = BitSim::Klines::path + symbol + ".dat";

    if (!std::filesystem::exists(file_path)) {
        return false;
    }

    const auto row_size = sizeof(time_point_ms) + sizeof(Kline::open_time) + sizeof(float);
    const auto row_count = std::filesystem::file_size(file_path) / row_size;

    auto data_file = std::ifstream{ file_path, std::ios::binary };

    data.try_emplace(symbol);
    auto&& entry = data[symbol];
    entry.resize(row_count);

    for (auto row_idx = 0; row_idx < row_count; row_idx++) {
        data_file.read(reinterpret_cast <char*> (&(entry[row_idx].open_time)), sizeof(Kline::open_time));
        data_file.read(reinterpret_cast <char*> (&(entry[row_idx].open_price)), sizeof(Kline::open_price));
        data_file.read(reinterpret_cast <char*> (&(entry[row_idx].volume)), sizeof(Kline::volume));
    }
    data_file.close();

    return true;
}
