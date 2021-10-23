#include "pch.h"
#include "Klines.h"

#include <fstream>


Klines::Klines(void)
{
    for (const auto& symbol : symbols) {
        printf("Load klines %s\n", symbol.name.data());
        load(symbol);
    }
}

bool Klines::load(const Symbol& symbol)
{
    const auto file_path = std::string{ BitSim::Klines::path } + symbol.name.data() + ".dat";

    if (!std::filesystem::exists(file_path)) {
        return false;
    }

    const auto row_count = std::filesystem::file_size(file_path) / sizeof(Kline);

    auto data_file = std::ifstream{ file_path, std::ios::binary };

    data[symbol.idx].clear();
    data_idx[symbol.idx] = 0;

    auto&& entry = data[symbol.idx];
    entry.resize(row_count);

    for (auto row_idx = 0; row_idx < row_count; row_idx++) {
        data_file.read(reinterpret_cast <char*> (&(entry[row_idx].open_time)), sizeof(Kline::open_time));
        data_file.read(reinterpret_cast <char*> (&(entry[row_idx].open)), sizeof(Kline::open));
        data_file.read(reinterpret_cast <char*> (&(entry[row_idx].high)), sizeof(Kline::high));
        data_file.read(reinterpret_cast <char*> (&(entry[row_idx].low)), sizeof(Kline::low));
        data_file.read(reinterpret_cast <char*> (&(entry[row_idx].volume)), sizeof(Kline::volume));
    }
    data_file.close();

    return true;
}

time_point_ms Klines::get_timestamp_start(void) const
{
    auto timestamp = data.begin()->front().open_time;
    for (const auto& symbol : symbols) {
        timestamp = std::max(timestamp, data[symbol.idx].front().open_time);
    }
    return timestamp;
}

time_point_ms Klines::get_timestamp_end(void) const
{
    auto timestamp = data.begin()->back().open_time;
    for (const auto& symbol : symbols) {
        timestamp = std::min(timestamp, data[symbol.idx].back().open_time);
    }
    return timestamp;
}

void Klines::step_idx(time_point_ms timestamp)
{
    for (const auto& symbol : symbols) {
        while (data[symbol.idx][data_idx[symbol.idx]].open_time < timestamp && data_idx[symbol.idx] + 1 < data[symbol.idx].size()) {
            data_idx[symbol.idx]++;
        }
    }
}

double Klines::get_open_price(const Symbol& symbol) const
{
    return data[symbol.idx][data_idx[symbol.idx]].open;
}

double Klines::get_high_price(const Symbol& symbol) const
{
    return data[symbol.idx][data_idx[symbol.idx]].high;
}

double Klines::get_low_price(const Symbol& symbol) const
{
    return data[symbol.idx][data_idx[symbol.idx]].low;
}

double Klines::get_volume(const Symbol& symbol) const
{
    return data[symbol.idx][data_idx[symbol.idx]].volume;
}
